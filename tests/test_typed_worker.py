"""A machine state whose worker types the loop's answer: two visits on both engines, a crash at
every op, and the two ways a visit stops.

Each visit notes something through the fallback, searches, appends to the bound workspace, and
cites a passage. The judge sends the first visit back. A visit whose answers are refused until
its repairs run out stops the run refused, and one cancelled on its repair stops it cancelled;
either way `committing` commits the tree from before that visit once."""

from enum import StrEnum
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from pydantic import BaseModel

from effective.api import Effect
from effective.cancel import Cancelled
from effective.channels import Field, Gated, Prompt, render
from effective.combinators import Level
from effective.domain import AskLLM, CallTool, DomainOp
from effective.keys import Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.outcomes import Advance, Exhausted, Finish, Park, ParkReason
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import agent_worker, build_specs
from effective.machine.trampoline import (
    RunCancelled,
    RunRefused,
    committing,
    run_machine,
    stop_record,
)
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

pytestmark = pytest.mark.conformance

type Passages = list[str]
LOG = "log.txt"
SEED = {LOG: "seed\n"}
AFTER_VISIT_0 = {LOG: "seed\nline 0\n"}
AFTER_VISIT_1 = {LOG: "seed\nline 0\nline 1\n"}
NEEDS = frozenset({"note", "search", "append"})
REFUSED = "The last answer was refused"
CITE = "cite a passage by its number"


class Work(StrEnum):
    WORK = "work"


class Verdict(StrEnum):
    GREEN = "green"
    RED = "red"


class Query(BaseModel):
    query: str


class AppendArgs(BaseModel):
    line: str


SEARCH = Tool("search", Query, list[str], observe=lambda passages: "\n".join(passages))
APPEND = Tool("append", AppendArgs, dict[str, str], observe=lambda tree: tree[LOG])


class GenerateAnswer(BaseModel):
    answer: str
    citation: int


def cited_answer(passages: Passages) -> Prompt[GenerateAnswer]:
    answer = Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), CITE)
    return render(t"Answer: {answer}\nCitation: {citation}", output=GenerateAnswer)


def acting(name: str, args: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(thought=name, tool=ToolRequest(name=name, args=args))


def answering(visit: str, citation: int) -> AssistantTurn:
    return AssistantTurn(thought="cite", answer={"answer": visit, "citation": citation})


class Visiting:
    """Each answer is a function of the request: the visit its prompt names, the tools run so
    far, and whether the tail carries a refusal. `bad` names the visits that never cite well,
    and `cancel` the visits whose repair is cancelled."""

    def __init__(self, bad: frozenset[int] = frozenset(), cancel: frozenset[int] = frozenset()):
        self.bad, self.cancel = bad, cancel
        self.asked = 0

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                self.asked += 1
                return self.turn(messages)
            case CallTool(name="note"):
                return ToolResult(content="noted")
            case CallTool(name="search"):
                return ["first", "second"]
            case CallTool(name="append", args={"line": line, "tree": tree}):
                return {**tree, LOG: tree.get(LOG, "") + line + "\n"}
            case CallTool(name="run_suite"):
                return CommandRun(exit_code=0)
            case _:
                raise AssertionError(f"unexpected op {op!r}")

    def turn(self, messages: list[dict[str, Any]]) -> Any:
        visit = messages[0]["content"]
        n = int(visit.removeprefix("visit "))
        ran = [m["content"] for m in messages if m["role"] == "tool"]
        repairing = REFUSED in messages[-1]["content"]
        match len(ran):
            case 0:
                return acting("note", {"text": visit})
            case 1:
                return acting("search", {"query": visit})
            case 2:
                return acting("append", {"line": f"line {n}", "tree": {"forged": "y"}})
            case _ if repairing and n in self.cancel:
                return Cancelled(partial="half")
            case _ if repairing and n not in self.bad:
                return answering(visit, 1)
            case _:
                return answering(visit, 7)


class Seen:
    def __init__(self) -> None:
        self.masked: list[frozenset[str]] = []
        self.read: list[tuple[GenerateAnswer, dict[str, str] | None]] = []

    def decide_for(self, needs: frozenset[str]) -> Decide:
        self.masked.append(needs)
        return default_decide

    def reader(self, trajectory: Trajectory[GenerateAnswer], log: ToolLog) -> Evidence:
        tree = log.last(dict[str, str])
        self.read.append((trajectory.answer, tree))
        return Evidence(summary=trajectory.answer.answer, tree=tree)


def noting(request: ToolRequest, level: Level) -> Effect[ToolResult]:
    return (yield from default_act(request, level))


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


def machine(seen: Seen):
    worker = agent_worker(
        prompt=lambda ctx: f"visit {ctx.visit}",
        answer=cited_answer,
        read=seen.reader,
        needs=NEEDS,
        decide_for=seen.decide_for,
        tools={"search": SEARCH, "append": APPEND},
        act=noting,
        bind=current,
    )
    return build_specs(Work, workers={Work.WORK: worker}, judges={Work.WORK: judge})


def visiting(run_id: str) -> Effect[dict[str, Any]]:
    seen = Seen()
    try:
        session = yield from committing(
            lambda: run_machine(
                Run(run_id),
                "cite",
                machine(seen),
                transition,
                start=Work.WORK,
                budget=3,
                tree=SEED,
            )
        )
    except (RunRefused, RunCancelled) as stopped:
        # The engine drives this body, so `pytest.raises` has no call to wrap.
        assert stopped.session.commitment is not None  # noqa: PT017
        return {
            "stopped": type(stopped).__name__,
            "tree": dict(stopped.tree),
            "files": list(stopped.session.commitment.files),
            "masked": len(seen.masked),
        }
    return {
        "path": list(session.path),
        "stopped": stop_record(session.stopped).kind,
        "read": [[answer.model_dump(), tree] for answer, tree in seen.read],
        "masked": [sorted(needs) for needs in seen.masked],
    }


def run_visiting(backend, domain: Visiting, fault: Fault):
    run_id = f"w{uuid4().hex}"
    name = compose_key(t"typed-worker:{Run(run_id)}").stored()
    backend.register(name, visiting, domain, fault, [])
    return run_id, backend.run_until_result(backend.spawn(name, run_id, max_attempts=80))


VISITED = {
    "path": ["work", "work"],
    "stopped": "machine-finished",
    "read": [
        [{"answer": "visit 0", "citation": 1}, AFTER_VISIT_0],
        [{"answer": "visit 1", "citation": 1}, AFTER_VISIT_1],
    ],
    "masked": [sorted(NEEDS), sorted(NEEDS)],
}


def test_a_typed_worker_runs_two_visits_on_both_engines(backend):
    _, snap = run_visiting(backend, Visiting(), Fault())
    assert snap.state == "completed", snap
    assert snap.result == VISITED


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_typed_worker_survives_a_crash_at_every_op_on_both_engines(backend, position):
    unarmed = Fault(position=position)
    _, snap = run_visiting(backend, Visiting(), unarmed)
    assert snap.state == "completed", snap
    assert unarmed.count == VISITING_OPS, "the walk changed shape; re-derive the count"
    for k, fault in at_every_op(unarmed):
        _, snap = run_visiting(backend, Visiting(), fault)
        assert snap.state == "completed", (k, snap)
        assert snap.result == VISITED, k


VISITING_OPS = 20
"""Each visit's note, search and append, each a decide and a call, then a refused answer and its
repair; and the postamble's four ops. The judge yields none."""


def test_a_visit_that_exhausts_its_repairs_stops_the_run_refused_on_both_engines(backend):
    run_id, snap = run_visiting(backend, Visiting(bad=frozenset({1})), Fault())
    assert snap.state == "completed", snap
    assert snap.result == {
        "stopped": "RunRefused",
        "tree": AFTER_VISIT_0,
        "files": [LOG],
        "masked": 2,
    }
    assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-parked"]


def test_a_visit_cancelled_on_its_repair_stops_the_run_cancelled_on_both_engines(backend):
    run_id, snap = run_visiting(backend, Visiting(cancel=frozenset({1})), Fault())
    assert snap.state == "completed", snap
    assert snap.result == {
        "stopped": "RunCancelled",
        "tree": AFTER_VISIT_0,
        "files": [LOG],
        "masked": 2,
    }
    assert backend.ledger_kinds(run_id) == ["machine-committed", "machine-parked"]


def test_a_typed_worker_without_a_reader_is_refused():
    with pytest.raises(ValueError, match="through `read`"):
        agent_worker(prompt=lambda ctx: ctx.goal, answer=cited_answer)  # ty: ignore[no-matching-overload]  -- the call the refusal is for
