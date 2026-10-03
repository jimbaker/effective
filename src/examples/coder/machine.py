"""The coder as a machine with one state: work, then let the test suite judge.

A visit is a ReAct loop over the four tools, and it ends when the model answers. The judge runs
the suite in the container over the tree the visit left: green finishes, red sends the next visit
back to work, and a suite that cannot run parks. The machine's postamble commits the tree and
records the suite's word on every ending.

A coder may delegate: its WORK state runs a child coder first and works on its goal with the
child's conclusion quoted beneath it. A child refused on the way is committed by the visit that
ran it, and the refusal climbs on.

A coder may carry skills: pins recorded once above the whole run, rendered into every visit's
prompt, and handed to each child with its delegation.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from enum import StrEnum
from functools import partial
from string.templatelib import Template
from typing import Any, assert_never

from effective.api import Effect, call_tool, scoped
from effective.channels import cite, skill
from effective.domain import ToolRefused
from effective.keys import Key, Name, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Advance, Exhausted, Finish, Outcome, Park, ParkReason
from effective.machine.spec import Ctx, Evidence, Worker
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import (
    SUITE_TOOL,
    Placement,
    committing,
    run_machine,
    running_under,
    stop_record,
)
from effective.react import ToolLog, Trajectory
from effective.skills import Pin
from examples.coder.tools import TOOLS, Changed


class Work(StrEnum):
    WORK = "work"


class Verdict(StrEnum):
    GREEN = "green"
    RED = "red"
    BROKEN = "broken"
    """The ENVIRONMENT could not run a suite, so nothing was measured.

    Reached when the suite tool refuses, which today means the container image is absent. A tree
    that does not collect is the model's own work and reads RED, because this state's job is to
    keep editing; `coding/verdicts.py` draws that line per state for the same reason."""


RETRY = (
    "\n\nAn earlier attempt ended while the tests were still failing. Run them to see why, then "
    "continue."
)


type Pins = Mapping[str, Pin]
"""Recorded skill pins by name, activated once above the run that renders them."""


def prompt(ctx: Ctx[Work], pins: Pins | None = None) -> Template:
    """The visit's task: the goal, the files there are, the pinned skills, and whether a visit
    already failed.

    The goal composes as it arrives. An operator's instruction is prose and renders as prose; a
    goal a parent built from what its child concluded is a `Template` whose quoted parts are
    already declared, and this hole splices it rather than flattening it. The file list is a data
    hole: a project names its own files, and `tree_path` admits a newline in one."""
    files = "\n".join(sorted(ctx.tree))
    retry = "" if ctx.incoming is None else RETRY
    guides = t""
    for name, pin in (pins or {}).items():
        guides += t"\n\n{skill(name, pin=pin)}"
    return t"{ctx.goal}\n\nProject files:\n{files:data}{guides}{retry}"


def workspace(ctx: Ctx[Work], log: ToolLog) -> dict[str, Any]:
    """The tree a tool call acts on: this visit's last recorded change, else its start."""
    changed = log.last(Changed)
    return {"tree": dict(ctx.tree if changed is None else changed.tree)}


def evidence(trajectory: Trajectory, log: ToolLog) -> Evidence:
    changed = log.last(Changed)
    return Evidence(summary=trajectory.answer, tree=None if changed is None else changed.tree)


def judge(ctx: Ctx[Work], evidence: Evidence) -> Effect[Verdict]:
    tree = ctx.tree if evidence.tree is None else evidence.tree
    schema: Any = ToolRefused | CommandRun
    run: ToolRefused | CommandRun = yield from call_tool(SUITE_TOOL, {"tree": dict(tree)}, schema)
    match run:
        case ToolRefused():
            return Verdict.BROKEN
        case CommandRun(collection_error=str()):
            # Asked explicitly rather than falling through the green test: `CommandRun` carries
            # this so a reader can tell "nothing ran" from "it ran and failed", and a state that
            # never asks cannot claim to have chosen.
            return Verdict.RED
        case CommandRun(green=True):
            return Verdict.GREEN
        case CommandRun():
            return Verdict.RED
        case unreachable:
            assert_never(unreachable)


def transition(state: Work, verdict: Verdict | Exhausted[Work]) -> Outcome[Work]:
    match verdict:
        case Verdict.GREEN:
            return Finish()
        case Verdict.RED:
            return Advance(Work.WORK)
        case Verdict.BROKEN:
            return Park(state, ParkReason.BROKEN_ENV)
        case Exhausted():
            return Park(state, ParkReason.EXHAUSTED)
        case unreachable:
            assert_never(unreachable)


def work_worker(turns: int = 12, pins: Pins | None = None) -> Worker[Work]:
    """The state's own work: a ReAct loop over the four tools, over the tree the visit was handed.

    Named so a caller can compose it. A worker that delegates to another machine wraps this one
    and keeps the loop it wraps identical to the one the plain coder runs."""
    return agent_worker(
        prompt=partial(prompt, pins=pins),
        tools=TOOLS,
        max_iters=turns,
        read=evidence,
        bind=workspace,
    )


def specs(
    turns: int = 12,
    worker: Worker[Work] | None = None,
    canonical: frozenset[Work] = frozenset(),
    pins: Pins | None = None,
):
    """The state map. `worker` composes over `work_worker`, and a worker that runs another machine
    inside this state reaches the canonical record from within it, so it declares `canonical`."""
    return build_specs(
        Work,
        workers={Work.WORK: worker or work_worker(turns, pins)},
        judges={Work.WORK: judge},
        canonical=canonical,
    )


DEEPER = compose_key(t"sub:{Name('deeper')}")

type Delegate = Callable[[Placement], Effect[dict[str, Any]]]
"""A child coder, handed where it will sit."""


def delegating(
    child: Delegate, under: Placement | None, turns: int = 12, pins: Pins | None = None
) -> Worker[Work]:
    """WORK that runs `child` first, then works on its goal with the child's conclusion cited.

    `under` is this run's own placement, which the child's extends by the state and visit running
    it. The child runs under `committing`, so a refused child is committed before its refusal
    climbs through this visit. The citation's address is the child's outcome row, as the child's
    run reported it."""
    inner = work_worker(turns, pins)

    def worker(ctx: Ctx[Work]) -> Effect[Evidence]:
        placed = running_under(under, ctx)
        below = yield from committing(lambda: scoped(DEEPER, lambda: child(placed)))
        concluded = cite(below["summary"], Key.parse(below["outcome"]))
        return (yield from inner(replace(ctx, goal=t"{ctx.goal}\nbelow: {concluded}")))

    return worker


def coder(
    run_id: str,
    goal: str | Template,
    tree: Mapping[str, str],
    *,
    visits: int = 3,
    turns: int = 12,
    under: Placement | None = None,
    delegate: Delegate | None = None,
    pins: Pins | None = None,
) -> Effect[dict[str, Any]]:
    """Work on `tree` toward `goal`, and report how the run stopped, what it committed, and what
    it concluded.

    `under` is where this run sits when another one is running it: two runs in one task compose one
    ledger address unless they say how they differ. `delegate` is a child coder each WORK visit
    runs first, and a state that runs one reaches the ledger from inside itself. `pins` render into
    every visit's prompt; a delegate that should carry them closes over them."""
    if delegate is None:
        worker, canonical = None, frozenset[Work]()
    else:
        worker, canonical = delegating(delegate, under, turns, pins), frozenset({Work.WORK})
    session = yield from run_machine(
        Run(run_id),
        goal,
        specs(turns, worker, canonical, pins),
        transition,
        start=Work.WORK,
        budget=visits,
        tree=tree,
        under=under,
    )
    stop = stop_record(session.stopped)
    assert session.commitment is not None, "the postamble always commits"
    last = session.concluded
    return {
        "stopped": stop.kind,
        "reason": stop.reason,
        "visits": len(session.turns),
        "passed": session.commitment.passed,
        "artifact_id": session.commitment.artifact_id,
        "summary": "" if last is None else last.summary,
        "outcome": session.outcome_id.stored(),
    }
