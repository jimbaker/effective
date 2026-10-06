"""A tool that takes the workspace in its arguments: `Ctx.tree`, `typed_act(bind=)` and
`agent_worker(bind=)`.

The tool here keeps nothing between calls, so each result is a function of the op's arguments
alone. What the tests pin is that those arguments carry the workspace the recorded results so far
describe, on both engines and across a crash at every op."""

from enum import StrEnum
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from pydantic import BaseModel

from effective.api import Effect
from effective.combinators import Level
from effective.domain import AskLLM, CallTool, DomainOp
from effective.handlers.base import TraceEntry
from effective.handlers.recording import RecordingHandler
from effective.keys import Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Advance, Exhausted, Finish, Park, ParkReason
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import run_machine, stop_record
from effective.ops import Step
from effective.react import AssistantTurn, Tool, ToolLog, ToolRequest, Trajectory, typed_act

LOG = "log.txt"


class Work(StrEnum):
    WORK = "work"


class Verdict(StrEnum):
    GREEN = "green"
    RED = "red"


class AppendArgs(BaseModel):
    line: str


APPEND = Tool("append", AppendArgs, dict[str, str], observe=lambda tree: tree[LOG])
TOOLS = {"append": APPEND}


def appended(tree: dict[str, str], line: str) -> dict[str, str]:
    return {**tree, LOG: tree.get(LOG, "") + line + "\n"}


class Stateless:
    """The model appends its prompt once per visit, then answers. The tool and the predicate
    answer from their arguments alone; the lists only record what arrived."""

    def __init__(self) -> None:
        self.bound: list[dict[str, str]] = []
        self.committed: list[dict[str, str]] = []

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages) if any(m["role"] == "tool" for m in messages):
                return AssistantTurn(thought="done", answer="appended")
            case AskLLM(messages=messages):
                line = messages[0]["content"]
                return AssistantTurn(
                    thought="append", tool=ToolRequest(name="append", args={"line": line})
                )
            case CallTool(name="append", args={"line": line, "tree": tree}):
                self.bound.append(tree)
                return appended(tree, line)
            case CallTool(name="run_suite", args={"tree": tree}):
                self.committed.append(tree)
                return CommandRun(exit_code=0)
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def current(ctx: Ctx, log: ToolLog) -> dict[str, Any]:
    return {"tree": log.last(dict[str, str]) or dict(ctx.tree)}


def read(trajectory: Trajectory, log: ToolLog) -> Evidence:
    return Evidence(summary=trajectory.answer, tree=log.last(dict[str, str]))


def judge(ctx: Ctx, evidence: Evidence) -> Effect[Verdict]:
    return Verdict.GREEN if ctx.visit == 1 else Verdict.RED
    yield  # pragma: no cover  -- a `Judge` is a generator


def transition(state: Work, verdict: Verdict | Exhausted[Work]):
    match verdict:
        case Verdict.GREEN:
            return Finish()
        case Verdict.RED:
            return Advance(Work.WORK)
        case Exhausted():
            return Park(state, ParkReason.EXHAUSTED)


SPECS = build_specs(
    Work,
    workers={
        Work.WORK: agent_worker(
            prompt=lambda ctx: f"line {ctx.visit}", tools=TOOLS, read=read, bind=current
        )
    },
    judges={Work.WORK: judge},
)


def appending(run_id: str) -> Effect[dict[str, Any]]:
    session = yield from run_machine(
        Run(run_id), "append", SPECS, transition, start=Work.WORK, budget=3, tree={LOG: "seed\n"}
    )
    return {"path": list(session.path), "stopped": stop_record(session.stopped).kind}


SEED = {LOG: "seed\n"}
AFTER_VISIT_0 = {LOG: "seed\nline 0\n"}
COMMITTED = {LOG: "seed\nline 0\nline 1\n"}

APPENDING_OPS = {FaultPosition.BEFORE_OP: 10, FaultPosition.AFTER_THUNK: 10}
"""Two visits of a decide, an append and an answering decide, then the postamble's four ops. The
judge yields none."""


def run_appending(backend, fault: Fault):
    run_id = f"a{uuid4().hex}"
    name = compose_key(t"bound-args:{Run(run_id)}").stored()
    domain = Stateless()
    backend.register(name, appending, domain, fault, [])
    return backend.run_until_result(backend.spawn(name, run_id)), domain


def test_each_visit_binds_the_tree_the_last_one_committed_on_both_engines(backend):
    snap, domain = run_appending(backend, Fault())
    assert snap.state == "completed", snap
    assert snap.result == {"path": ["work", "work"], "stopped": "machine-finished"}
    assert domain.bound == [SEED, AFTER_VISIT_0]
    assert domain.committed == [COMMITTED]


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_bound_tree_survives_a_crash_at_every_op_on_both_engines(backend, position):
    unarmed = Fault(position=position)
    snap, _ = run_appending(backend, unarmed)
    assert snap.state == "completed", snap
    assert unarmed.count == APPENDING_OPS[position], "the walk changed shape; re-derive the bound"
    for k, fault in at_every_op(unarmed):
        snap, domain = run_appending(backend, fault)
        assert snap.state == "completed", (k, snap)
        assert domain.committed[-1] == COMMITTED, k
        assert set(map(frozenset, (t.items() for t in domain.bound))) <= {
            frozenset(SEED.items()),
            frozenset(AFTER_VISIT_0.items()),
        }, k


def act_once(bind) -> CallTool:
    handler = RecordingHandler({"tool:append": {LOG: "x"}})
    act = typed_act(TOOLS, ToolLog(), bind=bind)
    forged = ToolRequest(name="append", args={"line": "a", "tree": {"forged": "y"}})
    handler.run(lambda: act(forged, Level(depth=0, model="", final=False)))
    match handler.trace:
        case [TraceEntry(op=Step(op=CallTool() as call))]:
            return call
        case trace:
            raise AssertionError(f"one tool call was expected: {trace}")


def test_without_bind_the_call_carries_the_models_arguments_unchanged():
    assert act_once(None).args == {"line": "a", "tree": {"forged": "y"}}


def test_a_bound_argument_overrides_the_models():
    assert act_once(lambda: {"tree": SEED}).args == {"line": "a", "tree": SEED}


def test_bind_without_tools_is_refused():
    with pytest.raises(ValueError, match="no `tools`"):
        agent_worker(prompt=lambda ctx: ctx.goal, bind=current)
