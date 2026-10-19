"""`engage` on the scripted AR world: an outside critic answers, and a critique must cite."""

import tempfile
from collections.abc import Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from effective.contexts import RecordingCtx
from effective.cost import Usage
from effective.domain import AskLLM, CallTool
from effective.govern import BudgetRefused, Exceeded
from effective.handlers.durable import DurableHandler
from examples.agent_firm import critic
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.critic import CLAIMED
from examples.agent_firm.engage import REPAIRS, Critique, CritiqueWire, asked, brief
from examples.agent_firm.world import World

CAPPED = {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025"], "reason": "4.2 caps"}


def _critic(answers: Iterator[Mapping[str, Any]]):
    def critic(op: Any) -> tuple[Any, Usage]:
        return CritiqueWire.model_validate(next(answers)), Usage()

    return critic


def _heard(*answers: Mapping[str, Any]) -> tuple[critic.Heard, World]:
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
    heard, world = _heard(*[answer] * (1 + REPAIRS))
    assert (heard.critique is not None) is held
    assert world.calls["critic"] == (1 if held else 1 + REPAIRS)


def test_a_refused_critique_is_asked_again_and_the_cited_one_stands():
    heard, world = _heard(CAPPED | {"cites": []}, CAPPED)
    assert critic.finds_the_cap(heard)
    assert world.calls["critic"] == 2


def test_the_critic_answers_and_the_firms_model_is_not_asked():
    _, world = _heard(CAPPED)
    assert set(world.calls) == {"critic"}


@pytest.mark.parametrize(
    ("change", "found"),
    [({}, True), ({"credit": "2500"}, False), ({"verdict": "accept"}, False)],
    ids=["the-cap", "full-hours-only", "accepted"],
)
def test_finding_the_cap_takes_a_revision_to_exactly_the_cap(change, found):
    heard, _ = _heard(CAPPED | change)
    assert critic.finds_the_cap(heard) is found


def test_a_banked_critique_replays_without_a_critic(tmp_path):
    world = World(critic=_critic(iter([CAPPED])), critiques=frozenset({CritiqueWire}))
    heard = critic.run(world, tmp_path / "task.db")
    assert critic.replay(tmp_path / "task.db") == heard


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
    critique = partial(asked, "engage", brief(SLA_CAP, CLAIMED), Critique, CritiqueWire)
    with pytest.raises(BudgetRefused):
        DurableHandler(RecordingCtx(), _Refusing()).run(critique)
