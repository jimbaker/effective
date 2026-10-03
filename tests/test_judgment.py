"""The `Judge` op: the t-string processor, `judge` and `select`, and each interpreter's arm."""

from datetime import date
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel

from effective import RecordingHandler, ReplayHandler, TraceEntry
from effective.api import judge, select
from effective.cost import BudgetExceeded, Contract, CostBudget, MeteredInterpreter, Usage
from effective.domain import (
    Answers,
    AskLLM,
    ChoiceAnswer,
    Judge,
    NoulAnswer,
    ScoreAnswer,
    WireChoice,
    WireNoul,
    WireScore,
)
from effective.handlers.absurd import DurableHandler
from effective.judgment import NO_MATCH, Choice, Noul, Score, TemplateError, Unasked, battery
from effective.ops import Step
from effective.sandbox import DryRun
from effective.sqlite import SqliteApp, SqliteTaskContext
from effective.telemetry import Span, traced

_TASK = UUID("019fa000-0000-7000-8000-000000000a01")

KIND = Choice({"guest": "someone the resident invited", "delivery": None})
EXPECTED = Noul(true="expected today")
SIZE = Score(["one person", "a couple", "a group"])


visitor = "Sam Ortiz"
caller = SimpleNamespace(name="Sam Ortiz", door="front door")
names = ["Samuel Ortiz", "Sam Ortiz-Reyes"]


@pytest.mark.parametrize(
    ("template", "state", "questions"),
    [
        pytest.param(
            t"Which kind? {KIND}",
            {},
            {"KIND": WireChoice(instructions="Which kind?", criteria=dict(KIND.criteria))},
            id="a question hole becomes a question named by its expression",
        ),
        pytest.param(
            t"The visitor {visitor} Expected? {EXPECTED}",
            {"visitor": "Sam Ortiz", "context": "The visitor `visitor`"},
            {"EXPECTED": WireNoul(instructions="Expected?", criteria={"true": "expected today"})},
            id="a state hole becomes a field and its literal text becomes context",
        ),
        pytest.param(
            t"{caller.name} came to the {caller.door} How many? {SIZE}",
            {
                "caller": {"name": "Sam Ortiz", "door": "front door"},
                "context": "`caller.name` came to the `caller.door`",
            },
            {"SIZE": WireScore(instructions="How many?", criteria=list(SIZE.levels))},
            id="a dotted state hole nests",
        ),
        pytest.param(
            t"{names} Same visitor? {EXPECTED:each names}",
            {"names": names, "context": "`names`"},
            {
                "EXPECTED[0]": WireNoul(
                    instructions="Same visitor? Judge `names[0]`.",
                    criteria={"true": "expected today"},
                ),
                "EXPECTED[1]": WireNoul(
                    instructions="Same visitor? Judge `names[1]`.",
                    criteria={"true": "expected today"},
                ),
            },
            id="an each hole asks once per element",
        ),
    ],
)
def test_the_processor_splits_state_from_questions(template, state, questions):
    made = battery(template)
    assert made.state == state
    assert made.questions == questions


@pytest.mark.parametrize(
    ("template", "refusal"),
    [
        pytest.param(t"{visitor} and nothing asked", "at least one question", id="no question"),
        pytest.param(t"{KIND} again {KIND}", "appears twice", id="a question twice"),
        pytest.param(t"{visitor} {visitor} {KIND}", "appears twice", id="a state field twice"),
        pytest.param(t"{KIND:every names}", "not implemented", id="an unknown spec"),
        pytest.param(t"{KIND:each missing}", "names no state field", id="each over nothing"),
        pytest.param(t"{visitor:!r} {KIND}", "not implemented", id="a state format spec"),
    ],
)
def test_the_processor_refuses_an_ambiguous_template(template, refusal):
    with pytest.raises(TemplateError, match=refusal):
        battery(template)


class Screened(BaseModel):
    KIND: ChoiceAnswer
    EXPECTED: NoulAnswer


class Merged(BaseModel):
    EXPECTED: list[NoulAnswer]


def _asked(recorder: RecordingHandler) -> Judge[Any]:
    """The one `Judge` a recorded run yielded."""
    match recorder.trace:
        case [TraceEntry(op=Step(op=Judge() as op))]:
            return op
        case other:
            raise AssertionError(f"expected one Judge step, got {other!r}")


def _choice(label: str, confidence: float = 0.9) -> dict[str, Any]:
    return {"choice": label, "confidence": confidence, "probabilities": {label: confidence}}


def _screen():
    return (
        yield from judge("screen", t"{visitor} Which kind? {KIND} Expected? {EXPECTED}", Screened)
    )


SCREENED = {"KIND": _choice("delivery"), "EXPECTED": {"p": 0.12}}


def test_judge_yields_one_judge_op_and_types_its_answers():
    recorder = RecordingHandler({"judge:screen": SCREENED})
    out = recorder.run(_screen)
    assert out == Screened(KIND=ChoiceAnswer(**_choice("delivery")), EXPECTED=NoulAnswer(p=0.12))
    assert _asked(recorder).state == {"visitor": "Sam Ortiz", "context": "`visitor`"}
    assert ReplayHandler(recorder.trace).run(_screen) == out


def test_an_each_question_answers_as_a_list_in_element_order():
    def merge():
        return (yield from judge("merge", t"{names} Same? {EXPECTED:each names}", Merged))

    answered = {"EXPECTED[1]": {"p": 0.2}, "EXPECTED[0]": {"p": 0.8}}
    out = RecordingHandler({"judge:merge": answered}).run(merge)
    assert out == Merged(EXPECTED=[NoulAnswer(p=0.8), NoulAnswer(p=0.2)])


def test_judge_refuses_an_output_whose_fields_are_not_the_questions():
    def mismatched():
        return (yield from judge("screen", t"{visitor} Which kind? {KIND}", Screened))

    with pytest.raises(TemplateError, match="fields"):
        RecordingHandler({}).run(mismatched)


def test_select_offers_the_candidates_and_no_match_only():
    def pick():
        return (
            yield from select(
                "resolve", t"{visitor} Which one is it?", ["Samuel Ortiz", "Alex Moreau"]
            )
        )

    recorder = RecordingHandler({"judge:resolve": {"pick": _choice(NO_MATCH)}})
    assert recorder.run(pick) == ChoiceAnswer(**_choice(NO_MATCH))
    assert _asked(recorder).questions["pick"].criteria == dict.fromkeys(
        ["Samuel Ortiz", "Alex Moreau", NO_MATCH]
    )


@pytest.mark.parametrize(
    "candidates",
    [
        pytest.param([], id="none"),
        pytest.param(["a", "a"], id="repeated"),
        pytest.param([NO_MATCH], id="no-match"),
    ],
)
def test_select_refuses_candidates_it_cannot_offer(candidates):
    def pick():
        return (yield from select("resolve", t"{visitor} Which?", candidates))

    with pytest.raises(TemplateError):
        RecordingHandler({}).run(pick)


def test_answers_validate_each_shape_as_exactly_one_model():
    raw = {
        "a": _choice("x"),
        "b": {"p": 0.5},
        "c": {"score": 2.0, "confidence": 0.7, "probabilities": {}},
    }
    assert [type(a) for a in Answers.model_validate(raw).root.values()] == [
        ChoiceAnswer,
        NoulAnswer,
        ScoreAnswer,
    ]


def _judgment() -> Judge[Answers]:
    """The op `_screen` yields: one choice and one noul over the visitor."""
    made = battery(t"{visitor} Which kind? {KIND} Expected? {EXPECTED}")
    return Judge(state=made.state, questions=made.questions, response_schema=Answers)


JEV_USAGE = Usage(prompt_tokens=400, cost=0.0000168)


def _judging(answers: dict[str, Any], usage: Usage = JEV_USAGE):
    asked: list[Judge[Any]] = []

    def call(op: Judge[Any]) -> tuple[Any, Usage]:
        asked.append(op)
        return Answers.model_validate(answers), usage

    return call, asked


def _poison(_op):
    raise AssertionError("replay must not reach the domain")


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def test_a_durable_judge_records_its_usage_and_replays_without_the_judge(app):
    call, asked = _judging(SCREENED)
    first = DurableHandler(
        SqliteTaskContext(app.conn, _TASK),
        MeteredInterpreter(llm=_poison, tools=_poison, judge=call),
        contract=Contract.V1,
    )
    out = first.run(_screen)
    assert out["KIND"]["choice"] == "delivery"  # a durable run returns its result as JSON
    assert len(asked) == 1
    assert first.meter == JEV_USAGE

    again = DurableHandler(
        SqliteTaskContext(app.conn, _TASK),
        MeteredInterpreter(llm=_poison, tools=_poison, judge=_poison),
        contract=Contract.V1,
    )
    assert again.run(_screen) == out
    assert again.meter == first.meter


def test_an_interpreter_with_no_judge_caller_says_so():
    interp = MeteredInterpreter(llm=_poison, tools=_poison)
    op = _judgment()
    with pytest.raises(LookupError, match="judge="):
        interp.run(op)


def test_a_spent_budget_refuses_the_next_judgment():
    call, asked = _judging({"KIND": _choice("delivery")}, Usage(prompt_tokens=10, cost=1.0))
    interp = MeteredInterpreter(
        llm=_poison, tools=_poison, budget=CostBudget(limit=1.0), judge=call
    )
    op = _judgment()
    interp.run(op)
    with pytest.raises(BudgetExceeded):
        interp.run(op)
    assert len(asked) == 1


def test_a_dry_run_forwards_a_judgment():
    call, asked = _judging({"KIND": _choice("delivery")})
    dry = DryRun(MeteredInterpreter(llm=_poison, tools=_poison, judge=call))
    dry.run(_judgment())
    assert len(asked) == 1
    assert dry.calls == []


def test_a_judgment_is_traced_as_a_model_call():
    spans: list[Span] = []
    call, _ = _judging({"KIND": _choice("delivery")})
    MeteredInterpreter(
        llm=_poison, tools=_poison, judge=call, domain_layers=[traced(spans.append)]
    ).run(_judgment())
    [span] = spans
    assert (span.name, span.kind, span.tool_name) == ("LM.0", "LLM", None)
    assert span.fields == {"effective.judge.questions": ["EXPECTED", "KIND"]}
    assert "Sam Ortiz" in span.input_messages[0]["content"]


@pytest.mark.parametrize(
    "answered",
    [
        pytest.param({"KIND": _choice("delivery")}, id="a question unanswered"),
        pytest.param({**SCREENED, "EXTRA": {"p": 0.5}}, id="an answer unasked"),
        pytest.param(
            {"KIND": _choice("rm -rf"), "EXPECTED": {"p": 0.1}}, id="a choice off the list"
        ),
        pytest.param({"KIND": {"p": 0.5}, "EXPECTED": {"p": 0.1}}, id="a noul for a choice"),
    ],
)
def test_an_answer_that_does_not_fit_its_question_is_refused(answered):
    with pytest.raises(Unasked):
        RecordingHandler({"judge:screen": answered}).run(_screen)


def test_select_refuses_a_bare_string_of_candidates():
    def pick():
        return (yield from select("resolve", t"{visitor} Which?", "Alex Moreau"))

    with pytest.raises(TemplateError, match="str"):
        RecordingHandler({}).run(pick)


def test_consecutive_model_calls_are_consecutive_turns():
    spans: list[Span] = []
    call, _ = _judging({"KIND": _choice("delivery")})
    interp = MeteredInterpreter(
        llm=lambda _op: ("ok", Usage()),
        tools=_poison,
        judge=call,
        domain_layers=[traced(spans.append)],
    )
    judged = _judgment()
    for op in (judged, judged, AskLLM(messages="hi", response_schema=str)):
        interp.run(op)
    assert len({span.span_id for span in spans}) == 3


def test_a_state_value_is_placed_as_json():
    when = date(2025, 12, 1)
    assert battery(t"{when} Expected? {EXPECTED}").state["when"] == "2025-12-01"
