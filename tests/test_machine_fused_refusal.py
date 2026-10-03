"""What a fused state's refused judge leaves behind, for each shape a state's run can take.

ROLE: adversarial. The walk runs a `Fused` phase by phase, so a refusal of its judge keeps the
tree its worker returned. Any other callable runs as written, and its judge's refusal keeps the
tree of the last visit that returned. Both engines, across replay and a crash at every op.
"""

from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition

from effective.api import Effect, append_ledger, ask_llm, call_tool
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import CallTool, DomainOp
from effective.govern import ChildRefused
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys import Name, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Exhausted, Finish, Outcome
from effective.machine.spec import Ctx, Evidence, Fused, Report, StateSpec
from effective.machine.spec import Run as StateRun
from effective.machine.specs import fuse
from effective.machine.trampoline import RunRefused, committing, run_machine
from effective.ops import LedgerRow


class Step(StrEnum):
    WORK = "work"


class Said(StrEnum):
    DONE = "done"


def finish(_state: Step, _verdict: Said | Exhausted[Step]) -> Outcome[Step]:
    return Finish()


STARTED = {"parent.py": "keep"}
WRITTEN = {"written.py": "new"}


def writing(_ctx: Ctx[Step]) -> Effect[Evidence]:
    """Writes through a recorded tool, so replay re-serves the tree."""
    return (yield from call_tool("write", {}, Evidence))


def refusing(_ctx: Ctx[Step], _evidence: Evidence) -> Effect[Said]:
    yield from call_tool("judge", {}, CommandRun)
    raise ChildRefused("judge refused")


def refusing_as_a_group(_ctx: Ctx[Step], _evidence: Evidence) -> Effect[Said]:
    yield from call_tool("judge", {}, CommandRun)
    raise ExceptionGroup(
        "judges", [ChildRefused("one"), ExceptionGroup("nested", [ChildRefused("two")])]
    )


def spending(_ctx: Ctx[Step], _evidence: Evidence) -> Effect[Said]:
    """Asks until the budget refuses."""
    for n in range(3):
        yield from ask_llm(f"judge-{n}", "spend", str)
    return Said.DONE


def accepting(_ctx: Ctx[Step], _evidence: Evidence) -> Effect[Said]:
    yield from call_tool("judge", {}, CommandRun)
    return Said.DONE


class Inherited(Fused):
    """A subclass that declares nothing of its own."""


class Overridden(Fused):
    """A subclass whose declared run is its own."""

    def __call__(self, ctx: Ctx[Step]) -> Effect[Report[Said]]:
        yield from append_ledger(
            LedgerRow(event_id=compose_key(t"note:{Name('override')}"), kind="override")
        )
        return Report(Said.DONE, summary="override ran", tree={"override.py": "kept"})


def forwarding(run: StateRun) -> StateRun:
    def forwarded(ctx: Ctx[Step]) -> Effect[Report[Said]]:
        return (yield from run(ctx))

    return forwarded


SHAPES: dict[str, tuple[Callable[[], StateRun], Mapping[str, str]]] = {
    "fused": (lambda: fuse(writing, refusing), WRITTEN),
    "a subclass": (lambda: Inherited(writing, refusing), STARTED),
    "a partial": (lambda: partial(fuse(writing, refusing)), STARTED),
    "a forwarding callable": (lambda: forwarding(fuse(writing, refusing)), STARTED),
}


class Tools:
    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="write"):
                return Evidence(summary="written", tree=WRITTEN)
            case _:
                return CommandRun(exit_code=0)


class Answers(Mapping[str, Any]):
    """The recorder's responses: `Tools`, by step name."""

    def __getitem__(self, key: str) -> Any:
        return (
            Evidence(summary="written", tree=WRITTEN)
            if "tool:write" in key
            else CommandRun(exit_code=0)
        )

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: object) -> bool:
        return True


def committed(run_id: str, run: StateRun, *, reraising: bool = False) -> Effect[dict[str, Any]]:
    """A caller committing the refused run through two `committing` frames."""
    specs = {Step.WORK: StateSpec(Step.WORK, run)}
    try:
        yield from committing(
            lambda: committing(
                lambda: run_machine(
                    Run(run_id), "go", specs, finish, start=Step.WORK, tree=STARTED
                )
            )
        )
    except RunRefused as refused:
        commitment = refused.session.commitment
        assert commitment is not None
        yield from call_tool("after", {}, CommandRun)
        if reraising:
            # The engine drives this body, so `pytest.raises` has no call to wrap.
            assert dict(refused.tree) == WRITTEN  # noqa: PT017
            raise
        return {"tree": dict(refused.tree), "files": list(commitment.files)}
    raise AssertionError("the run was not refused")


def registered(backend, body, domain: Any, fault: Fault | None = None, **budget: Any):
    run_id = f"f{uuid4().hex}"
    name = compose_key(t"fused-refusal:{Run(run_id)}").stored()
    backend.register(name, body, domain, fault or Fault(), [], **budget)
    task = backend.spawn(name, run_id, max_attempts=3, contract=Contract.V1)
    return run_id, task, backend.run_until_result(task)


@pytest.mark.parametrize("shape", SHAPES)
def test_each_shape_of_run_keeps_the_tree_its_walk_can_see(backend, shape):
    make, kept = SHAPES[shape]
    run_id, _task, snap = registered(backend, lambda rid: committed(rid, make()), Tools())

    assert snap.state == "completed", snap
    assert snap.result == {"tree": dict(kept), "files": list(kept)}
    assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-parked"]


def test_a_subclass_runs_the_call_it_declares(backend):
    run = Overridden(writing, accepting)

    def body(run_id: str) -> Effect[dict[str, Any]]:
        specs = {Step.WORK: StateSpec(Step.WORK, run, canonical=True)}
        session = yield from run_machine(Run(run_id), "go", specs, finish, start=Step.WORK)
        concluded, commitment = session.concluded, session.commitment
        assert concluded is not None
        assert commitment is not None
        return {"summary": concluded.summary, "files": list(commitment.files)}

    run_id, _task, snap = registered(backend, body, Tools())

    assert snap.state == "completed", snap
    assert snap.result == {"summary": "override ran", "files": ["override.py"]}
    assert backend.ledger_kinds(run_id) == ["override", "machine-committed", "machine-finished"]


JUDGES = {"bare": refusing, "grouped": refusing_as_a_group}


@pytest.mark.parametrize("reraising", [False, True], ids=["caught", "re-raised"])
@pytest.mark.parametrize("judge", JUDGES)
def test_replay_keeps_the_workers_tree(judge, reraising):
    body = partial(committed, "replayed", fuse(writing, JUDGES[judge]), reraising=reraising)
    recorder = RecordingHandler(Answers())
    if reraising:
        with pytest.raises(RunRefused) as recorded:
            recorder.run(body)
        with pytest.raises(RunRefused) as replayed:
            ReplayHandler(recorder.trace).run(body)
        assert replayed.value.tree == recorded.value.tree == WRITTEN
        assert type(replayed.value.__cause__) is type(recorded.value.__cause__)
    else:
        result = recorder.run(body)
        assert ReplayHandler(recorder.trace).run(body) == result
    assert [row.kind for row in recorder.ledger] == ["machine-committed", "machine-parked"]
    assert recorder.ledger[0].get("files") == ["written.py"]


SWEPT = {"bare": (refusing, 7), "grouped": (refusing_as_a_group, 7), "budget": (spending, 8)}
"""Each judge, and the ops one pass yields. A figure, recomputed by `Fault(position=…).count`."""


def metered() -> MeteredInterpreter:
    tools = Tools()
    return MeteredInterpreter(
        llm=lambda _op: ("spend", Usage(prompt_tokens=1, completion_tokens=1, cost=0.001)),
        tools=tools.run,
    )


def swept(backend, judge: Callable[..., Effect[Said]], fault: Fault) -> int:
    run_id, task, snap = registered(
        backend,
        partial(committed, run=fuse(writing, judge), reraising=True),
        metered(),
        fault,
        budget_limit=0.0015,
        on_exhaust="fail",
    )
    assert snap.state == "failed", snap
    assert "RunRefused" in str(snap.failure), snap.failure
    assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-parked"]
    return backend.task_attempts(task)


@pytest.mark.parametrize("position", list(FaultPosition))
@pytest.mark.parametrize("judge", SWEPT)
def test_a_refused_judge_survives_a_crash_at_every_op(backend, judge, position):
    """A refusal fails its task once, so each crash costs exactly one more attempt."""
    refuse, ops = SWEPT[judge]
    unarmed = Fault(position=position)
    assert swept(backend, refuse, unarmed) == 1
    assert unarmed.count == ops
    for k in range(1, ops + 1):
        fault = Fault(k, position=position)
        assert swept(backend, refuse, fault) == 2, k
        assert not fault.armed, k
