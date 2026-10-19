"""`agent_worker` over two visits: the masked decide, the declared tools with their bound
arguments, the fallback for an undeclared name, and the reader over the visit's own log.

Each visit notes something through the fallback, appends through the declared tool, then answers.
The judge sends the first visit back, so a second visit shows what each visit builds afresh."""

from enum import StrEnum
from typing import Any

from _scripted import Sequenced
from pydantic import BaseModel

from effective.api import Effect
from effective.combinators import Level
from effective.domain import CallTool
from effective.handlers.recording import RecordingHandler
from effective.keys import Run
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Advance, Exhausted, Finish, Park, ParkReason
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import Session, run_machine
from effective.ops import Step
from effective.react import (
    AssistantTurn,
    Decide,
    Tool,
    ToolLog,
    ToolRequest,
    ToolResult,
    Trajectory,
    default_act,
    default_decide,
)

LOG = "log.txt"
SEED = {LOG: "seed\n"}
NEEDS = frozenset({"append", "note"})


class Work(StrEnum):
    WORK = "work"


class Verdict(StrEnum):
    GREEN = "green"
    RED = "red"


class AppendArgs(BaseModel):
    line: str


APPEND = Tool("append", AppendArgs, dict[str, str], observe=lambda tree: tree[LOG])


def acting(name: str, args: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(thought=name, tool=ToolRequest(name=name, args=args))


def visit(n: int) -> list[AssistantTurn]:
    return [
        acting("note", {"text": f"visit {n}"}),
        acting("append", {"line": f"line {n}", "tree": {"forged": "by the model"}}),
        AssistantTurn(thought="done", answer=f"answered {n}"),
    ]


AFTER_VISIT_0 = {LOG: "seed\nline 0\n"}
AFTER_VISIT_1 = {LOG: "seed\nline 0\nline 1\n"}


class Visits:
    """What each seam of the worker saw, in order."""

    def __init__(self) -> None:
        self.masked: list[frozenset[str]] = []
        self.fallen_back: list[str] = []
        self.read: list[tuple[str, tuple[str, ...], dict[str, str] | None]] = []

    def decide_for(self, needs: frozenset[str]) -> Decide:
        self.masked.append(needs)
        return default_decide

    def fallback(self, request: ToolRequest, level: Level) -> Effect[ToolResult]:
        self.fallen_back.append(request.name)
        return (yield from default_act(request, level))

    def reader(self, trajectory: Trajectory, log: ToolLog) -> Evidence:
        tree = log.last(dict[str, str])
        self.read.append((trajectory.answer, log.names, tree))
        return Evidence(summary=trajectory.answer, tree=tree)


def current(ctx: Ctx, log: ToolLog) -> dict[str, Any]:
    return {"tree": log.last(dict[str, str]) or dict(ctx.tree)}


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


def driven() -> tuple[Session, Visits, RecordingHandler, Sequenced]:
    seen = Visits()
    specs = build_specs(
        Work,
        workers={
            Work.WORK: agent_worker(
                prompt=lambda ctx: f"visit {ctx.visit}",
                needs=NEEDS,
                decide_for=seen.decide_for,
                tools={"append": APPEND},
                act=seen.fallback,
                read=seen.reader,
                bind=current,
            )
        },
        judges={Work.WORK: judge},
    )
    table = Sequenced(
        {
            "react:turn": [*visit(0), *visit(1)],
            "tool:note": [ToolResult(content="noted"), ToolResult(content="noted")],
            "tool:append": [AFTER_VISIT_0, AFTER_VISIT_1],
            "tool:run_suite": [CommandRun(exit_code=0)],
        }
    )
    handler = RecordingHandler(table)
    session = handler.run(
        lambda: run_machine(
            Run("visits"), "append", specs, transition, start=Work.WORK, budget=3, tree=SEED
        )
    )
    assert isinstance(session, Session)
    return session, seen, handler, table


def calls(handler: RecordingHandler, name: str) -> list[dict[str, Any]]:
    return [
        entry.op.op.args
        for entry in handler.trace
        if isinstance(entry.op, Step) and isinstance(entry.op.op, CallTool)
        if entry.op.op.name == name
    ]


def test_two_visits_run_to_the_judges_verdict_and_consume_the_script():
    session, _, _, table = driven()
    assert (session.path, session.stopped) == ((Work.WORK, Work.WORK), Finish())
    assert table.unasked == {}


def test_each_visit_asks_the_decide_factory_with_the_states_repertoire():
    _, seen, _, _ = driven()
    assert seen.masked == [NEEDS, NEEDS]


def test_an_undeclared_name_goes_to_the_fallback_with_the_models_arguments():
    _, seen, handler, _ = driven()
    assert seen.fallen_back == ["note", "note"]
    assert calls(handler, "note") == [{"text": "visit 0"}, {"text": "visit 1"}]


def test_a_declared_tool_carries_the_bound_tree_over_the_models():
    _, _, handler, _ = driven()
    assert calls(handler, "append") == [
        {"line": "line 0", "tree": SEED},
        {"line": "line 1", "tree": AFTER_VISIT_0},
    ]


def test_the_reader_sees_only_its_visits_declared_results():
    _, seen, _, _ = driven()
    assert seen.read == [
        ("answered 0", ("append",), AFTER_VISIT_0),
        ("answered 1", ("append",), AFTER_VISIT_1),
    ]
