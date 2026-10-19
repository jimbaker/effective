"""The typed loop where it meets the rest of the loop: a redirect, a compaction, a cancel on the
final repair, a contract's parameters and directives, and the digest an untyped turn caches by."""

from typing import Any

import pytest
from _scripted import Sequenced
from pydantic import BaseModel

from effective import Effect
from effective.cache import op_digest
from effective.cancel import Cancelled
from effective.channels import ContractRefused, Field, Gated, Prompt, Repair, render
from effective.cost import Usage
from effective.domain import AskLLM, CallTool
from effective.handlers.recording import RecordingHandler
from effective.interpreters.tools import subagent_runner
from effective.interrupts import Interrupted, interrupt_key, tool_interrupt
from effective.ops import CompositionRefused
from effective.react import (
    AssistantTurn,
    Escaped,
    EvidenceUnbound,
    Exhausted,
    Tool,
    ToolRequest,
    ToolResult,
    TrajectorySummary,
    TurnRequest,
    UncarriedDirective,
    default_decide,
    run_agent,
    unwrap,
)

type Passages = list[str]
type Batch[T] = list[T]
PASSAGES = ["first", "second"]
CITE = "cite a passage by its number"
REFUSED = "The last answer was refused"


class Query(BaseModel):
    query: str


SEARCH = Tool("search", Query, list[str], observe=lambda passages: "\n".join(passages))


class Cited(BaseModel):
    answer: str
    citation: int


GOOD = Cited(answer="a", citation=1)


def cited(passages: Passages) -> Prompt[Cited]:
    answer = Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), CITE)
    return render(t"Answer: {answer}\nCitation: {citation}", output=Cited)


def searching() -> AssistantTurn:
    return AssistantTurn(thought="s", tool=ToolRequest(name="search", args={"query": "q"}))


def answering(citation: int) -> AssistantTurn:
    return AssistantTurn(thought="a", answer={"answer": "a", "citation": citation})


class Peek:
    """A decide that keeps each request it is handed."""

    def __init__(self) -> None:
        self.requests: list[TurnRequest] = []

    def __call__(self, request: TurnRequest) -> Effect[AssistantTurn]:
        self.requests.append(request)
        return (yield from default_decide(request))


def ran(table: dict[str, list[Any]], contract=cited, **kwargs: Any) -> tuple[Any, Peek, Sequenced]:
    peek, sequence = Peek(), Sequenced(table)
    ending = RecordingHandler(sequence).run(
        lambda: run_agent("cite", answer=contract, tools=[SEARCH], decide=peek, **kwargs)
    )
    return ending, peek, sequence


def test_a_redirect_on_a_repair_turn_carries_the_refusal_to_the_next_tail():
    quiet = Interrupted()
    ending, peek, sequence = ran(
        {
            "react:turn": [searching(), answering(7), answering(1)],
            "tool:search": [PASSAGES],
            interrupt_key("pre").stored(): [quiet, quiet, Interrupted(redirect="hurry"), quiet],
            interrupt_key("post").stored(): [quiet, quiet, quiet],
        },
        interrupt=tool_interrupt(),
    )
    assert unwrap(ending) == GOOD
    assert REFUSED in peek.requests[2].messages[-1]["content"]
    assert sequence.unasked == {}


def test_a_repair_turn_compacts_nothing():
    summary = TrajectorySummary()
    ending, _, sequence = ran(
        {
            "react:turn": [searching(), answering(7), answering(1)],
            "react:compact": [summary],
            "tool:search": [PASSAGES],
        },
        compact=lambda messages: any(m["role"] == "tool" for m in messages),
    )
    assert unwrap(ending) == GOOD
    assert sequence.unasked == {}


def test_a_turn_refused_for_acting_and_answering_leaves_the_action_out_of_the_transcript():
    both = AssistantTurn(
        thought="both", tool=ToolRequest(name="search", args={"query": "q"}), answer={}
    )
    _, peek, _ = ran({"react:turn": [searching(), both, answering(1)], "tool:search": [PASSAGES]})
    transcript = peek.requests[2].messages[:-1]
    assert transcript[-1] == {"role": "assistant", "content": "{}"}
    assert all("Action:" not in m["content"] for m in transcript[2:])


def test_a_cancel_on_the_final_repair_escapes():
    ending, _, _ = ran(
        {
            "react:turn": [searching()],
            "react:final": [answering(7), Cancelled(partial="half")],
            "tool:search": [PASSAGES],
        },
        max_iters=1,
    )
    assert isinstance(ending, Escaped)
    assert (ending.origin, ending.name, ending.partial) == ("decide", "react:final", "half")


def star(*passages: Passages) -> Prompt[Cited]:
    raise AssertionError("never called")


def keywords(**passages: Passages) -> Prompt[Cited]:
    raise AssertionError("never called")


def positional(passages: Passages, /) -> Prompt[Cited]:
    raise AssertionError("never called")


@pytest.mark.parametrize("contract", [star, keywords, positional], ids=lambda c: c.__name__)
def test_a_parameter_the_loop_cannot_pass_by_name_is_refused(contract):
    with pytest.raises(EvidenceUnbound) as refused:
        ran({}, contract=contract)
    assert (refused.value.parameter, refused.value.fault) == ("passages", "unpassable")


def test_a_parameter_with_a_default_receives_it_before_its_tool_answers():
    seen: list[Passages] = []

    def defaulted(passages: Passages = ["only"]) -> Prompt[Cited]:  # noqa: B006
        seen.append(passages)
        return cited(passages)

    ending, *_ = ran({"react:turn": [answering(0)]}, contract=defaulted)
    assert unwrap(ending) == Cited(answer="a", citation=0)
    assert seen[-1] == ["only"]


def test_a_declared_tool_whose_result_evidence_cannot_read_is_refused():
    # A generic alias is a form `Tool` declares and evidence cannot read; the refusal is the case.
    batched = Tool("batch", Query, Batch[str], observe=lambda batch: "")  # ty: ignore[invalid-argument-type]
    with pytest.raises(EvidenceUnbound) as refused:
        RecordingHandler({}).run(lambda: run_agent("x", answer=cited, tools=[SEARCH, batched]))
    assert refused.value.fault == "unsupported-result"


def test_a_role_in_the_contract_is_refused_as_the_tail_carries_none():
    def system(passages: Passages) -> Prompt[Cited]:
        answer, citation = Field(str), Field(int)
        return render(t"{'':role=system}Answer: {answer}\nCitation: {citation}", output=Cited)

    with pytest.raises(UncarriedDirective, match="declared a role"):
        ran({"react:turn": [searching()], "tool:search": [PASSAGES]}, contract=system)


def test_repairs_without_an_answer_are_refused():
    with pytest.raises(ValueError, match="no `answer`"):
        RecordingHandler({}).run(lambda: run_agent("x", repairs=3))  # ty: ignore[no-matching-overload]  -- the call the refusal is for


def test_a_refused_contract_fails_its_run_at_once():
    assert issubclass(ContractRefused, CompositionRefused)


def test_an_untyped_turns_cache_digest_is_pinned():
    """The digest covers the turn's response schema, so widening `AssistantTurn` re-keys every
    cached untyped turn. Pinned so the next change to it is deliberate."""
    asked = AskLLM(messages=[{"role": "user", "content": "hi"}], response_schema=AssistantTurn)
    assert op_digest(asked, "m") == UNTYPED_DIGEST


UNTYPED_DIGEST = "sha256-0fb8b86ef68a1b2c3cb05bb1d694ccf1b86ca3592caafff181a0abb598758ce9"


def test_a_subagent_that_exhausts_its_repairs_is_a_fenced_observation(monkeypatch):
    """The tool reads whatever ending its run stored, so each stored kind has its observation."""
    stored = Exhausted(repair=Repair(CITE), repairs=1, steps=[]).model_dump(mode="json")
    monkeypatch.setattr(
        "effective.interpreters.tools.DurableHandler.run", lambda self, program: stored
    )
    runner = subagent_runner(lambda op: (AssistantTurn(answer="x"), Usage()), {})
    observed = runner(CallTool(name="subagent", args={"task": "cite"}, result_schema=ToolResult))
    assert isinstance(observed, ToolResult)
    assert observed.content.startswith("[exhausted: repairs] ")
    assert CITE in observed.content


def test_a_redirect_before_the_final_level_carries_the_refusal_into_it():
    quiet = Interrupted()
    ending, peek, sequence = ran(
        {
            "react:turn": [searching(), answering(7)],
            "react:final": [answering(1)],
            "tool:search": [PASSAGES],
            interrupt_key("pre").stored(): [quiet, quiet, Interrupted(redirect="hurry")],
            interrupt_key("post").stored(): [quiet, quiet],
        },
        max_iters=2,
        interrupt=tool_interrupt(),
    )
    assert unwrap(ending) == GOOD
    assert peek.requests[-1].level.final
    assert REFUSED in peek.requests[-1].messages[-1]["content"]
    assert sequence.unasked == {}
