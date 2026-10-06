"""VOI probe policy layer: value-of-information probes that inform a budget grant.

Two products over the fork driver (`effective.fork`):

| product                                   | what it does                                        |
|-------------------------------------------|-----------------------------------------------------|
| informed grant prompt                     | at a park, shows the *free-and-exact* current best  |
| (`probe_prompt` -> `GrantPrompt`)         | against a *cheap probe* of one more level, each     |
|                                           | priced and graded; the human (or the rule           |
|                                           | `should_grant`) is the grader                       |
| informed-vs-blind policy (`probe_prompt`  | informed grants beat a fixed policy *only when      |
| + `should_grant` + `grant_full`)          | `probe ≪ grant`*; its grader is external (a `Gated` |
|                                           | verifier in production, `effective.channels`; a     |
|                                           | `Grader` callable in the offline batch)             |

Scope: forks a **workflow-yielded** count park (`descend`'s grant). A granted continuation
RE-runs from the park, with no fork promoted, so the informed policy's cost is
`probe + grant`; hence the condition `probe ≪ grant`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from effective.api import Effect
from effective.budget import (
    Grant,
    MeasuredBudget,
    budget_grant_name,
    depth_grant_name,
)
from effective.fork import ForkTail, MeasuredTail, MeteredEntry, OpIndex, fork_at, measured_drive
from effective.handlers.base import TraceEntry
from effective.keys import Key

type Grader = Callable[[Any], float]  # an answer -> quality in [0, 1]; a Gated verifier in prod
type Program[T] = Callable[[], Effect[T]]


@dataclass(frozen=True)
class Arm:
    """One arm of the grant prompt: its artifact, its post-park marginal cost (dollars), and
    the grader's quality on it."""

    label: str
    answer: Any
    cost: float
    quality: float


@dataclass(frozen=True)
class GrantPrompt:
    """The INFORMED grant prompt — the acceptance demo's artifact. `current_best` is
    free-and-exact (the recorded context, cost 0); `probe` is `probe_levels` drilled live, the
    cheap look-ahead that informs whether the (expensive) grant is worth it."""

    current_best: Arm
    probe: Arm

    @property
    def marginal_quality(self) -> float:
        return self.probe.quality - self.current_best.quality

    @property
    def marginal_cost(self) -> float:
        return self.probe.cost - self.current_best.cost


def probe_prompt[T](
    program: Program[T],
    base_trace: list[TraceEntry],
    run_id: str,
    depth: int,
    domain: Any,
    grader: Grader,
    *,
    probe_levels: int = 1,
) -> GrantPrompt:
    """Fork the parked run, drill `probe_levels` more (the cheap probe re-parks), and build the
    informed prompt: the free current best vs the probed look-ahead, each graded.

    **Generation 0, stated rather than defaulted.** `pending_name` has to name the park the base
    run actually made, and a `respawn` chain's park carries its generation — so a probe against a
    chained run would name generation 0's park and address nothing. VOI reads no generation today
    (nothing here touches `CHAIN_GENERATION`, and no caller respawns), so 0 is correct and this
    is the assumption written down at the three places that make it instead of hidden in one
    parameter default."""
    current = base_trace[-1].result  # the recorded current best (free-and-exact)
    tail: ForkTail = fork_at(
        program,
        base_trace,
        OpIndex(len(base_trace)),
        Grant(add_depth=probe_levels),
        domain,
        pending_name=depth_grant_name(run_id, depth=depth, generation=0),
    )
    probed = tail.trace[-1].result if tail.trace else current  # the deepest probed answer
    return GrantPrompt(
        current_best=Arm("stop", current, 0.0, grader(current)),
        probe=Arm(f"probe+{probe_levels}", probed, tail.usage.cost, grader(probed)),
    )


def should_grant(prompt: GrantPrompt, threshold: float) -> bool:
    """The rule tier: grant iff the probe's quality gain clears `threshold`. (The `human` tier
    reads the same prompt and decides; `should_grant` is the auto-answerer for the DoE.)"""
    return prompt.marginal_quality >= threshold


def grant_full[T](
    program: Program[T],
    base_trace: list[TraceEntry],
    run_id: str,
    depth: int,
    domain: Any,
    levels: int,
) -> ForkTail:
    """Fork the parked run and grant `levels` more, driving to a final answer (the finalize
    grant makes the last level final). The blind-grant arm and the informed policy's granted
    continuation both use this.

    Generation 0 for the same reason as `probe_prompt`."""
    finalize = depth_grant_name(run_id, depth=depth + levels, generation=0)
    return fork_at(
        program,
        base_trace,
        OpIndex(len(base_trace)),
        Grant(add_depth=levels),
        domain,
        pending_name=depth_grant_name(run_id, depth=depth, generation=0),
        grants={finalize: Grant(add_depth=0)},
    )


def measured_prompt[T](
    program: Program[T],
    prefix: list[MeteredEntry],
    budget: MeasuredBudget,
    domain: Any,
    grader: Grader,
    *,
    probe_dollars: float,
    delivered_grants: dict[Key, Grant] | None = None,
) -> GrantPrompt:
    """The DOLLARS informed grant prompt (slice B): fork a run parked at a *measured* ceiling,
    probe `probe_dollars` more of live spend, and price the marginal in dollars. `prefix` is the
    recorded measured trace up to the trip; `probe.cost` is the tail-only (`live_usage`) spend —
    the recorded prefix is replayed free.

    `delivered_grants` are the grants already delivered to *reach* this park (trips `0..K-1`); the
    pending trip is `K = len(delivered_grants)` (trips are sequential, so the name falls out as
    `budget-grant:{run_id}:{K}`). The probe fork must carry those prior grants, because
    `measured_drive` re-runs `enforce_measured` on every replayed `AskLLM` and would otherwise
    re-park at trip 0. Empty (the default) is the first park (K=0), the original single case."""
    delivered = delivered_grants or {}
    trip = budget_grant_name(budget.run_id, len(delivered))
    n = len(prefix)
    tail: MeasuredTail = measured_drive(
        program,
        budget,
        domain,
        {**delivered, trip: Grant(add_dollars=probe_dollars)},
        recorded=prefix,
    )
    # The current best is the decoded last prefix entry. Read it from the driver's trace (which
    # decodes each checkpoint AT THE OP), never off the raw bridged prefix entry: a bridged
    # `MeteredEntry` carries un-decoded `state` and no `.result`. The driver replays
    # the whole prefix before the probe, so `trace[n-1]` is that entry, decoded.
    current = tail.trace[n - 1].result if n else None
    probed = tail.trace[-1].result if tail.trace else current
    return GrantPrompt(
        current_best=Arm("stop", current, 0.0, grader(current)),
        probe=Arm(f"probe+${probe_dollars:g}", probed, tail.live_usage.cost, grader(probed)),
    )
