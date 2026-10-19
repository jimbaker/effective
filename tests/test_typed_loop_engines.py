"""The typed loop on both engines: a repair survives a crash at every op, a cancel on a repair's
own decide escapes, a budget refusal during a repair stays a refusal, and a spawned child that
exhausts its repairs is joined as a refusal."""

from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition, at_every_op
from _durable import IMMEDIATE_RETRY, absurd, pg_ready
from _gate import at_spend
from _spawning import absurd_spawner
from pydantic import BaseModel

from effective.api import Effect
from effective.budget import MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.cancel import Cancelled
from effective.channels import Field, Gated, Prompt, render
from effective.cost import MeteredInterpreter
from effective.domain import SPAWN_TOOL, AskLLM, CallTool, DomainOp
from effective.govern import BudgetRefused, ChildRefused, Proceed, govern
from effective.handlers.durable import DurableHandler
from effective.interpreters.tools import _adapt_ctx, make_tool_runner, spawn_tool
from effective.keys import Run, compose_key
from effective.ops import Step
from effective.react import (
    AssistantTurn,
    Escaped,
    Exhausted,
    Tool,
    ToolRequest,
    Trajectory,
    run_agent,
    unwrap,
)
from effective.spawning import join_answer, run_child, spawn_child

pytestmark = pytest.mark.conformance

type Passages = list[str]


class Query(BaseModel):
    query: str


SEARCH = Tool("search", Query, list[str], observe=lambda passages: "\n".join(passages))
REFUSED = "The last answer was refused"
CITE = "cite a passage by its number"


class GenerateAnswer(BaseModel):
    answer: str
    citation: int


def cited_answer(passages: Passages) -> Prompt[GenerateAnswer]:
    answer = Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), CITE)
    return render(t"Answer: {answer}\nCitation: {citation}", output=GenerateAnswer)


class Citing:
    """The model searches, cites a passage that does not exist, then cites one that does. Each
    answer is a function of the request, so a turn asked again after a crash answers alike."""

    def __init__(self, repair: Any = None) -> None:
        self.repair = repair
        self.asked: list[str] = []
        self.searches = 0

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                tail = messages[-1]["content"]
                if not any(m["role"] == "tool" for m in messages):
                    self.asked.append("search")
                    return AssistantTurn(
                        thought="s", tool=ToolRequest(name="search", args={"query": "q"})
                    )
                if REFUSED not in tail:
                    self.asked.append("bad")
                    return AssistantTurn(thought="a", answer={"answer": "a", "citation": 7})
                self.asked.append("repair")
                if self.repair is not None:
                    return self.repair
                return AssistantTurn(thought="a", answer={"answer": "a", "citation": 1})
            case CallTool(name="search"):
                self.searches += 1
                return ["first", "second"]
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def citing(run_id: str) -> Effect[dict[str, Any]]:
    ending = yield from run_agent("find and cite", answer=cited_answer, tools=[SEARCH])
    match ending:
        case Trajectory(answer=answer):
            return {"answer": answer.model_dump(), "repairs": [s.repair for s in ending.steps]}
        case Escaped(origin=origin, name=name, partial=partial):
            return {"escaped": [origin, name, partial]}
        case Exhausted(repairs=repairs):
            return {"exhausted": repairs}


def budgeted(run_id: str) -> Effect[dict[str, Any]]:
    try:
        return (yield from citing(run_id))
    except BudgetRefused as refused:
        return {"budget": refused.reason}


def run_citing(backend, workflow, domain, fault: Fault, layers=()):
    run_id = f"t{uuid4().hex}"
    name = compose_key(t"typed-loop:{Run(run_id)}").stored()
    backend.register(name, workflow, domain, fault, list(layers))
    return backend.run_until_result(backend.spawn(name, run_id, max_attempts=50))


ANSWERED = {"answer": {"answer": "a", "citation": 1}, "repairs": [None, CITE, None]}

CITING_OPS = {FaultPosition.BEFORE_OP: 4, FaultPosition.AFTER_THUNK: 4}
"""A search decide, the search, the refused answer and its repair: four ops."""


def test_a_repaired_run_answers_on_both_engines(backend):
    domain = Citing()
    snap = run_citing(backend, citing, domain, Fault())
    assert snap.state == "completed", snap
    assert snap.result == ANSWERED
    assert domain.asked == ["search", "bad", "repair"]


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_a_repair_survives_a_crash_at_every_op_on_both_engines(backend, position):
    unarmed = Fault(position=position)
    assert run_citing(backend, citing, Citing(), unarmed).state == "completed"
    assert unarmed.count == CITING_OPS[position], "the walk changed shape; re-derive the count"
    for k, fault in at_every_op(unarmed):
        domain = Citing()
        snap = run_citing(backend, citing, domain, fault)
        assert snap.state == "completed", (k, snap)
        assert snap.result == ANSWERED, k
        rerun = 1 if position == FaultPosition.AFTER_THUNK else 0
        assert len(domain.asked) + domain.searches <= len(ANSWERED["repairs"]) + 1 + rerun, k


def test_a_cancel_on_the_repairs_own_decide_escapes_on_both_engines(backend):
    domain = Citing(repair=Cancelled(partial="half"))
    snap = run_citing(backend, citing, domain, Fault())
    assert snap.state == "completed", snap
    assert snap.result == {"escaped": ["decide", "react:turn", "half"]}
    assert domain.asked == ["search", "bad", "repair"]


def test_a_budget_refusal_during_a_repair_stays_a_refusal_on_both_engines(backend):
    over = at_spend(
        0.01, budget_policy(MeasuredBudget(overall=0.005, run_id="r", on_exhaust="fail"))
    )

    def the_repair(op, state):
        match op:
            case Step(op=AskLLM(messages=messages)) if REFUSED in messages[-1]["content"]:
                return over(op, state)
            case _:
                return Proceed()

    layers = [govern(the_repair, gate="spend", run_id="r")]
    domain = Citing()
    snap = run_citing(backend, budgeted, domain, Fault(), layers)
    assert snap.state == "completed", snap
    assert set(snap.result) == {"budget"}
    assert domain.asked == ["search", "bad"]


def asks_nothing(op: AskLLM[Any]) -> Any:
    raise AssertionError("the parent asks no model")


class Miscites(Citing):
    """Every answer cites a passage that does not exist."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages) if any(m["role"] == "tool" for m in messages):
                return AssistantTurn(thought="a", answer={"answer": "a", "citation": 7})
            case _:
                return super().run(op)


@pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")
def test_a_spawned_child_that_exhausts_its_repairs_is_joined_as_a_refusal_with_its_reason():
    app, spawner_app = absurd(), absurd()
    try:
        mid = f"exhausted-{uuid4().hex[:8]}"
        pname, cname = f"p-{mid}", f"c-{mid}"

        @app.register_task(cname)
        def child(params, ctx):
            ctx = _adapt_ctx(ctx)
            handler = DurableHandler(ctx, Miscites(), params=params)

            def cites() -> Effect[GenerateAnswer]:
                return unwrap((yield from run_agent("cite", answer=cited_answer, tools=[SEARCH])))

            return run_child(ctx, params, handler, cites)

        @app.register_task(pname)
        def parent(params, ctx):
            def hears() -> Effect[str]:
                spawned = yield from spawn_child(cname, mid, {})
                try:
                    yield from join_answer(spawned)
                except ChildRefused as refused:
                    return str(refused)
                return "answered"

            spawner = absurd_spawner(spawner_app, IMMEDIATE_RETRY, [])
            tools = make_tool_runner({}, agents={SPAWN_TOOL: spawn_tool(spawner)})
            return DurableHandler(ctx, MeteredInterpreter(llm=asks_nothing, tools=tools)).run(
                hears
            )

        snap = app.run_until_result(app.spawn(pname, {"mid": mid}))
        assert snap is not None
        assert snap.state == "completed", f"state={snap.state} failure={snap.failure}"
        assert snap.result == f"RepairsExhausted: the answer was refused after 1 repairs: {CITE}"
    finally:
        app.close()
        spawner_app.close()
