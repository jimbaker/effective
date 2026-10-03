"""Counterfactual safety: what a fork may observe, and what it may never touch.

A counterfactual that writes is a second run. The guarantee has two halves, and they land in two
different places, which is the point worth reading before adding a third:

1. **Suppress the canonical commit.** On the DURABLE path this seam already exists and is not a
   layer: `DurableHandler(ledger=None)` checkpoints an `AppendLedgerRow` and appends nothing
   (`handlers/absurd.py::_record_ledger`). An op-layer would be wrong there: a layer that
   answers `AppendLedgerRow`/`StoreArtifact` without forwarding collapses the
   checkpoint sequence from `['tool:a', 'ledger:…', 'artifact:…']` to `['tool:a']`, destroying the
   parity a fork's diff depends on and un-persisting the artifact (pre-CAS the checkpoint IS the
   artifact store). In-process there is no durable substrate to commit to, so the drivers here
   answer those ops directly: the same guarantee by a different mechanism, because the two paths
   have different correct behavior.
2. **Stop the world.** Neither `ledger=None` nor an in-process driver stops a *tool* from mutating
   the world; only a dry-run domain does. That is `DryRun` below, and it is the half nothing else
   covered.

The observable contract both halves are held to, which is what a conformance test should assert
(per-op decisions legitimately differ by driver):

    the canonical ledger is untouched · the world is untouched · the workflow sees the same values
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, assert_never

from effective.cost import Usage
from effective.counterfactual import ForkedDeadline as ForkedDeadline  # re-export
from effective.counterfactual import ForkedSleep as ForkedSleep  # re-export: see its docstring
from effective.domain import AsksModel, CallTool, DomainOp
from effective.handlers.base import artifact_id
from effective.keys import Key
from effective.ops import (
    AppendLedgerRow,
    AwaitEvent,
    Scoped,
    SleepUntil,
    StoreArtifact,
    WorkflowOp,
)

# --- half 1: what an in-process driver does with a non-metered op ------------------------


@dataclass(frozen=True)
class Observed:
    """The value an inspect-only op returns into the workflow — as if it had really run."""

    value: Any


@dataclass(frozen=True)
class Unanswered:
    """No answer was delivered for an await, so the fork parks rather than inventing one.

    Named for the CAUSE: `budget.Parked` already means "the measured trip parked", and the
    refusal-name family keeps one spelling per meaning. A driver decides what an unanswered await
    means for it: the in-process fork reports a park; a durable child would suspend."""

    name: Key


type Inspection = Observed | Unanswered


class ForkPointRefused(RuntimeError):
    """A fork was taken AT an op that is not a decision — refused.

    `fork_at` substitutes a counterfactual result for the op at the fork point; that is only
    meaningful for a **decision** — a `Step`'s result or an `AwaitEvent`'s answer. Substituting a
    `SleepUntil` (a clock — see `ForkedSleep`), an `AppendLedgerRow`, or a `StoreArtifact` (writes,
    not decisions) would fabricate a counterfactual for an op that carries no decision. Fork at the
    decision the write depends on, not at the write."""

    def __init__(self, op: object) -> None:
        super().__init__(
            f"cannot fork at a {type(op).__name__}: a fork substitutes a DECISION (a Step result "
            f"or an AwaitEvent answer), not a write. Fork at the decision, not its consequence."
        )
        self.op = op


class ForkPointInsideScope(ForkPointRefused):
    """A fork was taken at an op that IS a decision, but one running inside a ``scoped(...)``
    body — refused, for a reason that has nothing to do with what the op is.

    Its own class because the parent's message is actively wrong here. That one says a fork
    substitutes *"a Step result or an AwaitEvent answer"* — and this refusal fires on exactly
    those, so a reader was told the op they forked at is the legal kind while being refused for
    forking at it. The op is fine; the REGION is the problem.

    **Why the region is.** `replay_prefix` drives a scoped body as a sub-generator of its own
    rather than one the workflow ``yield from``-ed, so a fork point inside it names an op whose
    parent frame the replay cannot hand back — there is no generator to return positioned at it.
    A closed scope in the prefix forks fine; it is the OPEN one containing the fork point that
    cannot be resumed.

    The same shape as `GatherRegionFork` one module over: a region that is one node in the fork's
    order, refused with the rule rather than with a guess about the op."""

    def __init__(self, op: object, scope: str) -> None:
        RuntimeError.__init__(
            self,
            f"cannot fork at a {type(op).__name__} inside the scope {scope!r}: the op is a "
            f"decision, but a scoped body is driven as a sub-generator, so replay cannot hand "
            f"back a generator positioned at an op inside it. Fork at a decision BEFORE the "
            f"scope or AFTER it — a scope is one node in the fork's order.",
        )
        self.op = op
        self.scope = scope


def inspect_only(op: WorkflowOp, answers: Mapping[Key, Any]) -> Inspection:
    """Interpret one non-metered op WITHOUT committing anything — the shared policy.

    ONE definition, driven by every in-process driver (`measured_drive`, `live_drive`), so a
    counterfactual's safety rules cannot drift between them — the same "one transition, many
    drivers" move as `enforce_measured`, `permission.decide` and `govern.combine`.

    - `AppendLedgerRow` — **not appended.** Returns `None`, exactly as a real append does, so the
      workflow's control flow is identical while the canonical record is untouched. A fork that
      appended would forge history for a run that never happened (two bookkeepers).
    - `StoreArtifact` — **not stored.** The id is content-addressed, so it can be *derived* from
      the value; the fork gets the id the real run would, for free. (On the DURABLE path this op
      must still checkpoint — the checkpoint is the artifact store there. Different driver,
      different correct behaviour, same observable contract.)
    - `SleepUntil` — **refused** (`ForkedSleep`). A counterfactual explores a decision, and a
      sleep inside one is a category error to refuse.
    - `AwaitEvent` naming no deadline — answered from `answers` if one was delivered, else
      `Unanswered`.
    - `AwaitEvent` naming one — **refused** (`ForkedDeadline`), by `ForkedSleep`'s argument: a
      grant says what arrived, and a deadline asks whether it arrived in time, which is a
      question about elapsed time the fork does not spend. The deadline is DESTRUCTURED in both
      arms rather than ignored in one, so a wait the fork cannot answer is a missing arm here
      instead of a dropped field.
    - anything else — refused loudly. Guessing at a `Gather` inside a counterfactual would
      silently change what is being measured.
    """
    match op:
        case AppendLedgerRow():
            return Observed(None)
        case StoreArtifact():
            return Observed(artifact_id(op))
        case SleepUntil(when=when):
            raise ForkedSleep(when)
        case AwaitEvent(name=name, deadline=None):
            return Observed(answers[name]) if name in answers else Unanswered(name)
        case AwaitEvent(name=name, deadline=datetime() as deadline):
            raise ForkedDeadline(name, deadline)
        case Scoped():
            raise TypeError(
                "a Scoped must never reach `inspect_only`: it is STRUCTURE, and the drivers "
                "interpret it themselves (recursing under the extended path) rather than asking "
                "what a counterfactual may observe. Reaching here means a driver grew a "
                "`case _` that swallowed it — add the `Scoped` arm there."
            )
        case _:
            raise TypeError(
                f"a counterfactual cannot interpret {type(op).__name__}: the in-process fork "
                f"drivers replay one sequential schedule and cannot interpret a concurrent op. "
                f"Fork a sequential prefix, or use the durable driver (`run_fork`), which runs a "
                f"`Gather` in its forked tail."
            )


# --- half 2: the dry-run domain — the half `ledger=None` does not cover ------------------


class WorldMutation(RuntimeError):
    """A counterfactual tried to call a tool that is not known to be read-only.

    Raised rather than mocked, deliberately. A fabricated result would let the fork report a
    marginal computed from an answer nobody produced; a refusal says the counterfactual cannot be
    run as posed. World-mutating tools are not reversible by forking, so the fork stops."""

    def __init__(self, tool: str, allowed: Iterable[str]) -> None:
        known = ", ".join(sorted(allowed)) or "(none)"
        super().__init__(
            f"a fork may not call the tool {tool!r}: it is not in the read-only allow-list "
            f"({known}). A counterfactual must not mutate the world — add {tool!r} to `allow` if "
            f"it is genuinely read-only, or supply a canned result for it."
        )
        self.tool = tool


@dataclass
class DryRun:
    """A domain interpreter a counterfactual can safely run: model calls yes, world writes no.

    `allow` is an **allow-list of read-only tool names** — the fail-safe direction, since a
    forgotten entry stops the fork loudly while a forgotten *deny* entry would let it write.
    `canned` supplies results for tools that should answer without being called at all.

    A model call, `AskLLM` or `Judge`, always forwards: it has no world effect beyond cost,
    and a counterfactual that never asks the model measures nothing. Cost is the budget's job,
    so a fork runs under a `MeasuredBudget`, and the sandbox holds no spend rule.

    Wraps any domain exposing `run` / `run_metered` and preserves BOTH, so it drops in under the
    metered path without hiding `run_metered` (the `serve` lesson: a `.run`-only wrapper silently
    disables the cap)."""

    base: Any
    allow: frozenset[str] = frozenset()
    canned: Mapping[str, Any] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def _guard(self, op: DomainOp[Any]) -> Any | None:
        """The gate: `None` means "forward it", anything else is the answer to return."""
        match op:
            case AsksModel():
                return None
            case CallTool(name=name):
                self.calls.append(name)
                if name in self.canned:
                    return self.canned[name]
                if name in self.allow:
                    return None
                raise WorldMutation(name, self.allow)
            case unreachable:
                # A gate whose unknown case forwards fails open, so a new op is a type error here.
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead

    def run(self, op: DomainOp[Any]) -> Any:
        answer = self._guard(op)
        return self.base.run(op) if answer is None else answer

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        answer = self._guard(op)
        if answer is None:
            return self.base.run_metered(op)
        return answer, Usage()  # a canned answer cost nothing — the meter must not claim it did


__all__ = [
    "DryRun",
    "ForkPointInsideScope",
    "ForkPointRefused",
    "ForkedSleep",
    "Inspection",
    "Observed",
    "Unanswered",
    "WorldMutation",
    "inspect_only",
]
