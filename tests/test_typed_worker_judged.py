"""A typed worker's visit judged by Jev: the state's judge asks Jev whether the cited passage
grounds the answer, and sends the visit back when it does not.

The worker types the loop's answer, and the reader hands the judge the passage the answer cites.
The judge yields a `Judge` op, which a `Jev` interpreter serves over a fake client, so the two
grains run together: the typed loop inside the visit, a Jev judgment on its evidence."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pytest

sdk = pytest.importorskip("typesafe_sdk")

from pydantic import BaseModel  # noqa: E402

from effective.api import Effect, judge  # noqa: E402
from effective.channels import Field, Gated, Prompt, render  # noqa: E402
from effective.contexts import RecordingCtx, ReplayCtx  # noqa: E402
from effective.cost import MeteredInterpreter, Usage  # noqa: E402
from effective.domain import CallTool, ChoiceAnswer  # noqa: E402
from effective.handlers.durable import DurableHandler  # noqa: E402
from effective.interpreters.jev import Jev  # noqa: E402
from effective.interpreters.scripted import scripted_caller  # noqa: E402
from effective.judgment import Choice  # noqa: E402
from effective.keys import Run  # noqa: E402
from effective.machine.evidence import CommandRun  # noqa: E402
from effective.machine.outcomes import Advance, Exhausted, Finish, Park, ParkReason  # noqa: E402
from effective.machine.spec import Ctx, Evidence  # noqa: E402
from effective.machine.specs import agent_worker, build_specs  # noqa: E402
from effective.machine.trampoline import run_machine, stop_record  # noqa: E402
from effective.react import AssistantTurn, Tool, ToolLog, ToolRequest, Trajectory  # noqa: E402
from effective.spend import TokenBudget  # noqa: E402

PASSAGES = ["the moon is cheese", "the moon is rock"]
GROUNDED = Choice({"grounded": "the passage states the answer", "ungrounded": None})


class Work(StrEnum):
    WORK = "work"


class Verdict(StrEnum):
    GREEN = "green"
    RED = "red"


class Query(BaseModel):
    query: str


SEARCH = Tool("search", Query, list[str], observe=lambda passages: "\n".join(passages))


class Cited(BaseModel):
    answer: str
    citation: int


def cited(passages: list[str]) -> Prompt[Cited]:
    answer = Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), "cite a passage by its number")
    return render(t"Answer: {answer}\nCitation: {citation}", output=Cited)


def read(trajectory: Trajectory[Cited], log: ToolLog) -> Evidence:
    passages = log.last(list[str]) or []
    return Evidence(summary=trajectory.answer.answer, detail=passages[trajectory.answer.citation])


class Review(BaseModel):
    GROUNDED: ChoiceAnswer


def judged(ctx: Ctx, evidence: Evidence) -> Effect[Verdict]:
    passage = evidence.detail
    review = yield from judge("review", t"{passage} Does it ground the answer? {GROUNDED}", Review)
    return Verdict.GREEN if review.GROUNDED.choice == "grounded" else Verdict.RED


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
            prompt=lambda ctx: "what is the moon?",
            answer=cited,
            read=read,
            tools={"search": SEARCH},
        )
    },
    judges={Work.WORK: judged},
)


def reviewing(run_id: str) -> Effect[dict[str, Any]]:
    session = yield from run_machine(
        Run(run_id), "answer", SPECS, transition, start=Work.WORK, budget=3
    )
    return {"path": list(session.path), "stopped": stop_record(session.stopped).kind}


@dataclass
class Reviewer:
    """A Jev client that calls a passage grounded when it says the moon is rock."""

    asked: list[dict[str, Any]] = field(default_factory=list)

    def system_one(self, *, state, questions, model):
        self.asked.append(state)
        label = "grounded" if "rock" in state["passage"] else "ungrounded"
        return sdk.SystemOneResponse(
            model=model,
            usage=sdk.Usage(input_tokens=400, output_tokens=None),
            answers={
                "GROUNDED": sdk.ChoiceAnswer(
                    choice=label, confidence=0.9, probabilities={label: 0.9}
                )
            },
        )


U = Usage(prompt_tokens=8, completion_tokens=3, cost=0.001)
SEARCHING = AssistantTurn(thought="s", tool=ToolRequest(name="search", args={"query": "moon"}))


def answering(citation: int) -> AssistantTurn:
    return AssistantTurn(thought="a", answer={"answer": "the moon", "citation": citation})


def tools(op: CallTool[Any]) -> Any:
    match op.name:
        case "search":
            return PASSAGES
        case "run_suite":
            return CommandRun(exit_code=0)
        case name:
            raise AssertionError(f"unexpected tool {name}")


def test_jev_sends_back_a_typed_visit_whose_citation_does_not_ground_its_answer(tmp_path):
    reviewer = Reviewer()
    interpreter = MeteredInterpreter(
        llm=scripted_caller(
            [(SEARCHING, U), (answering(0), U), (SEARCHING, U), (answering(1), U)]
        ),
        tools=tools,
        judge=Jev(reviewer, TokenBudget(tmp_path / "ledger", 10_000)),
    )
    ctx = RecordingCtx()
    result = DurableHandler(ctx, interpreter).run(lambda: reviewing("judged"))
    assert result == {"path": ["work", "work"], "stopped": "machine-finished"}
    assert [state["passage"] for state in reviewer.asked] == PASSAGES

    silent = MeteredInterpreter(
        llm=scripted_caller([]), tools=tools, judge=Jev(Reviewer(), TokenBudget(tmp_path / "x", 0))
    )
    assert DurableHandler(ReplayCtx(ctx.log), silent).run(lambda: reviewing("judged")) == result
