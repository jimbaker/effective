"""Assembling a spec map, and the ReAct loop as a worker — generic over the state type.

`build_specs` iterates the state type it is handed rather than one it imports, which is what
keeps it generic over embodiments.
"""

from collections.abc import Callable, Mapping
from enum import StrEnum
from functools import partial
from string.templatelib import Template
from typing import Any

from effective.machine.evidence import CommandRun, Measured
from effective.machine.spec import (
    Ctx,
    Evidence,
    Fused,
    Judge,
    Run,
    StateSpec,
    Worker,
)
from effective.react import Act, Decide, Tool, ToolLog, Trajectory, run_agent, typed_act


def build_specs[S: StrEnum, V: StrEnum, R: Measured = CommandRun](
    states: type[S],
    *,
    workers: Mapping[S, Worker[S, R]],
    judges: Mapping[S, Judge[S, Any, R]],
    default_worker: Worker[S, R] | None = None,
    default_judge: Judge[S, Any, R] | None = None,
    canonical: frozenset[S] = frozenset(),
) -> dict[S, StateSpec[S, Any, R]]:
    """Assemble a TOTAL spec map, iterating `State` rather than the caller's keys.

    That direction is the whole point. Building the map from `workers`' keys would produce whatever
    the caller remembered; building it from `State` means a state the caller forgot is a loud
    failure here, at assembly, instead of a `KeyError` inside the trampoline on the one walk that
    happens to reach it.

    `default_worker`/`default_judge` cover the states a deployment has not filled in. Passing
    neither a specific nor a default entry for some state is refused — with the states named,
    because "your map is incomplete" without the list is a puzzle."""
    missing = [
        state.value
        for state in states
        if (state not in workers and default_worker is None)
        or (state not in judges and default_judge is None)
    ]
    if missing:
        raise ValueError(
            f"no worker/judge for {', '.join(sorted(missing))} and no default supplied — a spec "
            f"map must be total over `State`, because the trampoline indexes it directly"
        )
    return {
        state: StateSpec(
            state=state,
            run=fuse(
                workers.get(state) or _required(default_worker),
                judges.get(state) or _required(default_judge),
            ),
            canonical=state in canonical,
        )
        for state in states
    }


def fuse[S: StrEnum, V: StrEnum, R: Measured](
    worker: Worker[S, R], judge: Judge[S, V, R]
) -> Run[S, V, R]:
    """The two-phase state, as a CONSTRUCTION over the one-slot primitive.

    An embodiment can decline it, and fusing is the right default: most states judge by computing a
    verdict from a measurement, which is one step, not two places on the map.

    **Declining the fuse is about needing an ADDRESS, not about needing coordinates.** Both forms
    hand the judgment a `Ctx`, so "my judge must author a canonical `event_id`" is answered here
    without splitting anything. Write the judgment as a state of its own when it wants what only
    a state has: its own line in the budget, its own edge in the transition table, or a name a
    skill pack can bind to.

    The two forms compose in one embodiment, and the artifact is
    `tests/test_machine_fuse_or_split.py`: WORK fuses while REVIEW and RULE are a split pair, both
    halves append canonical rows, and all four ids stay distinct."""

    return Fused(worker, judge)


def _required[T](value: T | None) -> T:
    assert value is not None, "the totality check above proves this unreachable"
    return value


def agent_worker(
    *,
    prompt: Callable[[Ctx], str | Template],
    needs: frozenset[str] = frozenset(),
    decide_for: Callable[[frozenset[str]], Decide] | None = None,
    tools: Mapping[str, Tool[Any, Any]] | None = None,
    act: Act | None = None,
    max_iters: int = 6,
    read: Callable[[Trajectory, ToolLog], Evidence] | None = None,
    bind: Callable[[Ctx, ToolLog], Mapping[str, Any]] | None = None,
) -> Worker:
    """A state's worker as a `run_agent` loop: the two grains composed, as the package
    docstring's diagram draws them.

    **The repertoire is masked through `decide_for`.** `run_agent` has no tool-catalog parameter
    and should not grow one: the allowed set is a property of how a turn is DECIDED, and the
    caller that constrains a decode is the model caller, which lives deployment-side. So this
    takes a factory from a repertoire to a `Decide`; a deployment plugs in a bracketed caller, a
    test plugs in a scripted one, and the substrate never learns a vendor's name.

    **`tools` is what makes the loop compose with a real table.** Without it every action is
    checkpointed as a `ToolResult`, one string field, so a tool returning the
    resulting workspace fails validation on a durable engine, and a state's edits never reach the
    commit. With it, `typed_act` yields each tool's declared result schema and records what came
    back. The masking question is untouched: `tools` says what a name MEANS, `decide_for` says
    what a turn may name, and the two are different seams on purpose.

    `read` turns the loop's output into `Evidence`, and it is a parameter because the map is
    domain knowledge: which result carried the workspace, whether one of them was a measurement.
    It takes the `ToolLog` as well as the `Trajectory` because the trajectory keeps only
    `observation: str`, so a reader given it alone could recover a tree only by parsing back a
    rendered string. The default reports the answer and nothing else: correct for a state with no
    mechanical judge, and deliberately too little for one with a mechanical judge, which
    `mechanical_judges` then names.

    `bind` reads the visit's `Ctx` and the results so far into the arguments `typed_act` merges
    into each tool call, such as the workspace a stateless tool edits. It binds only `tools`, so
    passing it without them is refused."""
    if bind is not None and not tools:
        raise ValueError("`bind` merges arguments into `tools`' calls, and no `tools` were given")
    to_evidence = read or _answer_only

    def worker(ctx: Ctx) -> Any:
        decide = decide_for(needs) if decide_for is not None else None
        # PER INVOCATION, not per `agent_worker` call. This closure is re-entered on every visit
        # to the state, so a log built beside `to_evidence` above would carry visit 0's edits
        # into visit 3's evidence — and a self-edge (`STILL_RED` -> DRAFT) makes that the
        # ordinary case rather than an exotic one.
        log = ToolLog()
        bound = partial(bind, ctx, log) if bind is not None else None
        dispatch = typed_act(tools, log, fallback=act, bind=bound) if tools else act
        trajectory = yield from run_agent(prompt(ctx), max_iters, decide=decide, act=dispatch)
        return to_evidence(trajectory, log)

    return worker


def _answer_only(trajectory: Trajectory, log: ToolLog) -> Evidence:
    """The default reader: the answer, and how it stopped.

    No `measured` and no `tree`, which are the two fields a replay depends on being derived from
    RECORDED results. It ignores the `ToolLog` deliberately rather than folding it: which result
    is a workspace and which is a measurement is an EMBODIMENT's fact, and a substrate default
    that guessed would be right for the coding table and quietly wrong for the next one.
    `effective.coding.specs.coding_read` is the fold for the table that ships."""
    return Evidence(
        summary=trajectory.answer,
        detail=f"{len(trajectory.steps)} steps, stopped: {trajectory.stop_reason}",
    )


__all__ = [
    "agent_worker",
    "build_specs",
    "fuse",
]
