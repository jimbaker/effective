"""`engage` on the scripted AR world: an outside critic answers, and a critique must cite."""

import json
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from effective.contexts import RecordingCtx
from effective.cost import RecordedFormat, Usage
from effective.domain import AskLLM, CallTool
from effective.govern import BudgetRefused, Exceeded
from effective.handlers.durable import DurableHandler
from examples.agent_firm import critic, deduction_review
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.critic import AS_CLAIMED
from examples.agent_firm.engage import REPAIRS, Critique, CritiqueWire, asked, brief
from examples.agent_firm.world import World

CAPPED = {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025"], "reason": "4.2 caps"}


def _critic(answers: Iterator[Mapping[str, Any]]):
    def critic(op: Any) -> tuple[Any, Usage]:
        return CritiqueWire.model_validate(next(answers)), Usage()

    return critic


def _reviewed(*answers: Mapping[str, Any]) -> tuple[critic.Review, World]:
    world = World(critic=_critic(iter(answers)), critiques=frozenset({CritiqueWire}))
    with tempfile.TemporaryDirectory() as directory:
        return critic.run(world, Path(directory) / "task.db"), world


@pytest.mark.parametrize(
    ("cites", "held"),
    [([], False), (["MSA-NW-2025"], True), (["MSA-NW-2025", "MSA-XX-0000"], False)],
    ids=["none", "held", "one-not-held"],
)
def test_a_critique_counts_only_when_every_citation_is_held(cites, held):
    answer = CAPPED | {"cites": cites}
    review, world = _reviewed(*[answer] * (1 + REPAIRS))
    assert (review.critique is not None) is held
    assert world.calls["critic"] == (1 if held else 1 + REPAIRS)


def test_a_refused_critique_is_asked_again_and_the_cited_one_stands():
    review, world = _reviewed(CAPPED | {"cites": []}, CAPPED)
    assert critic.finds_the_cap(review)
    assert world.calls["critic"] == 2


def test_the_critic_answers_and_the_firms_model_is_not_asked():
    _, world = _reviewed(CAPPED)
    assert set(world.calls) == {"critic"}


@pytest.mark.parametrize(
    ("change", "found"),
    [({}, True), ({"credit": "2500"}, False), ({"verdict": "accept"}, False)],
    ids=["the-cap", "full-hours-only", "accepted"],
)
def test_finding_the_cap_takes_a_revision_to_exactly_the_cap(change, found):
    review, _ = _reviewed(CAPPED | change)
    assert critic.finds_the_cap(review) is found


def test_a_banked_critique_replays_without_a_critic(tmp_path):
    world = World(critic=_critic(iter([CAPPED])), critiques=frozenset({CritiqueWire}))
    review = critic.run(world, tmp_path / "task.db")
    with closing(sqlite3.connect(tmp_path / "task.db")) as db:
        ((params,),) = db.execute("SELECT params FROM tasks").fetchall()
    assert RecordedFormat.from_params(json.loads(params)) is RecordedFormat.V1
    assert critic.replay(tmp_path / "task.db") == review


def test_a_model_ask_and_a_tool_of_one_name_count_apart():
    world = World(llm=lambda op: {}, tools={"CritiqueWire": lambda args: {"ok": True}})
    domain = world.domain()
    domain.run(AskLLM(messages="", response_schema=CritiqueWire))
    domain.run(CallTool(name="CritiqueWire", args={}, result_schema=dict))
    assert (world.calls, world.asked) == ({"llm": 1, "CritiqueWire": 1}, {"CritiqueWire": 1})


class _Refusing:
    """A domain whose budget refuses every op's spend."""

    def run(self, op: Any) -> Any:
        raise BudgetRefused(op, Exceeded(1, 1))


def test_a_refused_spend_is_raised_out_of_a_repair():
    """The critic's spend refused is no channel's refusal, so it is not asked again."""
    critique = partial(asked, "engage", brief(SLA_CAP, AS_CLAIMED), Critique, CritiqueWire)
    with pytest.raises(BudgetRefused):
        DurableHandler(RecordingCtx(), _Refusing()).run(critique)


def test_every_deduction_review_banked_from_a_live_model_replays_and_finds_the_cap():
    runs = deduction_review.replayed()
    assert {name: critic.finds_the_cap(review) for name, review in runs.items()} == {
        "haiku/0": True,
        "haiku/1": True,
        "haiku/2": True,
    }


def test_a_deduction_review_replays_from_the_dump_it_was_banked_as(tmp_path):
    world = World(critic=_critic(iter([CAPPED])), critiques=frozenset({CritiqueWire}))
    assert deduction_review.banked(world, tmp_path / "scripted" / "0")
    assert set(deduction_review.replayed(tmp_path)) == {"scripted/0"}
