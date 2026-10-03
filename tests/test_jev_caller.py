"""The Jev caller: the wire both ways, kept answers, and the budget refused before a call."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

sdk = pytest.importorskip("typesafe_sdk")

from effective.domain import (  # noqa: E402
    ChoiceAnswer,
    Judge,
    NoulAnswer,
    ScoreAnswer,
    WireChoice,
    WireNoul,
    WireScore,
)
from effective.interpreters.aio import shared_loop  # noqa: E402
from effective.interpreters.jev import AsyncJev, Jev  # noqa: E402
from effective.spend import TokenBudget, TokenBudgetExhausted  # noqa: E402

JUDGED = Judge(
    state={"visitor": "Sam Ortiz"},
    questions={
        "kind": WireChoice(instructions="Which kind?", criteria={"delivery": None, "guest": "x"}),
        "expected": WireNoul(instructions="Expected?", criteria={"true": "expected"}),
        "size": WireScore(instructions="How big?", criteria=["small", "large"]),
    },
    response_schema=dict,
)


@dataclass
class FakeClient:
    """Answers every question it is asked and counts the calls."""

    asked: list[dict[str, Any]] = field(default_factory=list)

    def system_one(self, *, state, questions, model):
        self.asked.append({"state": state, "questions": questions, "model": model})
        return sdk.SystemOneResponse(
            model=model,
            usage=sdk.Usage(input_tokens=400, output_tokens=None),
            answers={
                "kind": sdk.ChoiceAnswer(
                    choice="delivery", confidence=0.8, probabilities={"delivery": 0.8}
                ),
                "expected": sdk.NoulAnswer(noul=0.1),
                "size": sdk.ScoreAnswer(
                    score=1.0, confidence=0.6, legend={}, probabilities={0: 0.4, 1: 0.6}
                ),
            },
        )


@dataclass
class AsyncFakeClient(FakeClient):
    """`FakeClient`, awaited, noting the thread each call ran on."""

    threads: list[str] = field(default_factory=list)
    loops: list[Any] = field(default_factory=list)

    async def system_one(self, *, state, questions, model):
        self.threads.append(threading.current_thread().name)
        self.loops.append(asyncio.get_running_loop())
        return super().system_one(state=state, questions=questions, model=model)


FLAVORS = ["sync", "async"]


def _client(flavor: str) -> FakeClient:
    return FakeClient() if flavor == "sync" else AsyncFakeClient()


def _jev(flavor: str, client: Any, budget: TokenBudget, **kept: Any) -> Jev:
    return Jev(client, budget, **kept) if flavor == "sync" else AsyncJev(client, budget, **kept)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_jev_converts_the_wire_both_ways_and_meters_input_tokens(tmp_path, flavor):
    client = _client(flavor)
    budget = TokenBudget(tmp_path / "spend.json", 1_000_000)
    answers, usage = _jev(flavor, client, budget)(JUDGED)
    [asked] = client.asked
    assert isinstance(asked["questions"]["expected"], sdk.Noul)
    assert answers.root == {
        "kind": ChoiceAnswer(choice="delivery", confidence=0.8, probabilities={"delivery": 0.8}),
        "expected": NoulAnswer(p=0.1),
        "size": ScoreAnswer(score=1.0, confidence=0.6, probabilities={"0": 0.4, "1": 0.6}),
    }
    assert (usage.prompt_tokens, budget.spent) == (400, 400)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_kept_answer_is_served_free_and_asks_nothing(tmp_path, flavor):
    client = _client(flavor)
    budget = TokenBudget(tmp_path / "spend.json", 1_000_000)
    jev = _jev(flavor, client, budget, cache=tmp_path / "kept")
    first, _ = jev(JUDGED)
    again, usage = jev(JUDGED)
    assert again == first
    assert (len(client.asked), usage.prompt_tokens, budget.spent) == (1, 0, 400)


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_spent_budget_refuses_before_the_call_and_binds_across_instances(tmp_path, flavor):
    ledger = tmp_path / "spend.json"
    TokenBudget(ledger, 400).add(400)
    client = _client(flavor)
    with pytest.raises(TokenBudgetExhausted):
        _jev(flavor, client, TokenBudget(ledger, 400))(JUDGED)
    assert client.asked == []


def test_an_unreadable_kept_answer_is_asked_again(tmp_path):
    client = FakeClient()
    jev = Jev(client, TokenBudget(tmp_path / "spend", 10**6), cache=tmp_path / "kept")
    jev(JUDGED)
    [kept] = (tmp_path / "kept").iterdir()
    kept.write_text('{"answers": ')
    answers, _ = jev(JUDGED)
    assert len(client.asked) == 2
    assert answers.root["expected"] == NoulAnswer(p=0.1)


def test_callers_keeping_one_answer_at_once_all_succeed(tmp_path):
    client = FakeClient()
    jev = Jev(client, TokenBudget(tmp_path / "spend", 10**6), cache=tmp_path / "kept")
    failures: list[BaseException] = []

    def ask() -> None:
        try:
            jev(JUDGED)
        except BaseException as err:
            failures.append(err)

    callers = [threading.Thread(target=ask) for _ in range(8)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join()
    assert failures == []
    assert list((tmp_path / "kept").glob("*.partial")) == []


def test_the_async_judge_is_awaited_on_the_loop_and_answered_to_a_synchronous_caller(tmp_path):
    client = AsyncFakeClient()
    answers, _ = AsyncJev(client, TokenBudget(tmp_path / "spend", 10**6))(JUDGED)
    assert client.threads == ["interpreter-loop"]
    assert client.loops == [shared_loop().loop], "the judge ran on the process's one loop"
    assert threading.current_thread().name != "interpreter-loop"
    assert answers.root["expected"] == NoulAnswer(p=0.1)
