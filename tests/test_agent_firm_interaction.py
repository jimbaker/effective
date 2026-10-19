"""Knowledge and a critic, two by two, as an instrument check on both engines: four runs, one per
combination of snapshot (with the SLA rule or without) and critic (present or absent), each in a
world and books of its own. The interaction is how much more the critic moves the outcome with the
rule held than without.

The worlds are scripted, so any interaction is what a script put there: these pins test the
instrument, never complementarity.

| property                     | what holds                                                    |
|------------------------------|---------------------------------------------------------------|
| no interaction               | a null world, where no model reads the rule, measures none    |
| additive effects             | the rule and the critic each taking $1,000 off measures none  |
| swapped labels flip the sign | labeling the snapshot without the rule treated flips the sign |
| a missing outcome is refused | a dropped, doubled or stray outcome is refused, and named     |
| read off the books           | an interaction reads 0 off completion, $2,000 off the books   |
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from _firm import briefed, manifest, resolved

from effective.cost import Usage
from examples.agent_firm.ar import Round
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.engage import CritiqueWire, Proposal
from examples.agent_firm.factors import interaction
from examples.agent_firm.knowledge import NO_RULES, WITH_SLA_RULE, Pinned
from examples.agent_firm.manifest import Manifest
from examples.agent_firm.score import scored

KNOWN, RULED = (
    Pinned(height=0, digest=NO_RULES.digest),
    Pinned(height=1, digest=WITH_SLA_RULE.digest),
)
"""The firm's founding, `NO_RULES`, and the snapshot holding the SLA rule."""

type Credit = Callable[[bool], int]
"""What a model answers, by whether its brief holds the SLA rule."""


def _world(proposes: Credit, critiques: Credit):
    firm = briefed()

    def proposer(op: Any) -> Any:
        credit = proposes("POL-SLA-1" in str(op.messages))
        return Proposal(credit=Decimal(credit), rationale="r", cites=("CRM-2207",))

    def critic(op: Any) -> tuple[Any, Usage]:
        credit = critiques("POL-SLA-1" in str(op.messages))
        critique = {"verdict": "revise", "credit": str(credit), "cites": ["MSA-NW-2025"]}
        return CritiqueWire.model_validate(critique | {"reason": "r"}), Usage()

    firm.world.llm, firm.world.critic = proposer, critic
    return firm


WORLDS: dict[str, tuple[Credit, Credit]] = {
    "null": (lambda ruled: 3000, lambda ruled: 1000),
    "additive": (lambda ruled: 2000 if ruled else 3000, lambda ruled: 1000 if ruled else 2000),
    "interacting": (lambda ruled: 3000, lambda ruled: 1000 if ruled else 3000),
}
"""Each world's proposer and critic, as the credit each answers with the rule in hand or not."""


def _arms(backend, world: str) -> list[tuple[Manifest, Any, Decimal, bool]]:
    """Each arm's manifest, its result, the outcome read off the books as dollars from the gold
    credit, and the outcome read off whether its run completed."""
    books, arms = f"x-{uuid4().hex[:8]}", []
    for pinned in (KNOWN, RULED):
        for critic in (False, True):
            firm = _world(*WORLDS[world])
            arm = f"k{pinned.height}a{int(critic)}"
            admission = manifest(pinned, Round(f"{books}-{arm}"), critic=critic, arm=arm)
            run = resolved(backend, firm, admission, hypothetical=True)
            error = scored(SLA_CAP, firm.world.erp).error
            arms.append((admission, run, -error, run.snapshot.state == "completed"))
    return arms


@pytest.mark.parametrize("world", ["null", "additive"])
def test_a_world_without_interaction_measures_none(backend, world):
    arms = _arms(backend, world)
    assert interaction([(admission, y) for admission, _, y, _ in arms], KNOWN, RULED) == 0


def test_an_interacting_world(backend):
    arms = _arms(backend, "interacting")
    books = [(admission, y) for admission, _, y, _ in arms]
    completed = [(admission, Decimal(done)) for admission, _, _, done in arms]
    assert interaction(books, KNOWN, RULED) == Decimal(2000)
    assert interaction(books, RULED, KNOWN) == Decimal(-2000)
    assert interaction(completed, KNOWN, RULED) == 0
    with pytest.raises(ValueError, match="no outcome"):
        interaction(books[1:], KNOWN, RULED)
    with pytest.raises(ValueError, match="two outcomes"):
        interaction([*books, books[0]], KNOWN, RULED)
    stray = Pinned(height=2, digest="another-snapshot")
    strays = [(admission.model_copy(update={"knowledge": stray}), y) for admission, y in books]
    with pytest.raises(ValueError, match="neither arm"):
        interaction(books[2:] + strays[:2], KNOWN, RULED)
