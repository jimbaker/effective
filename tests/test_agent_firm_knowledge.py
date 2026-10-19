"""The firm's knowledge K as a stock: a snapshot each run reads at a recorded step, advanced by an
adoption claimed on the ledger, on both engines.

| question | the pin                                                                            |
|----------|------------------------------------------------------------------------------------|
| C1       | an old run's replay holds after the head advances                                  |
| C2       | a no-op delta leaves both briefs byte-identical                                    |
| C3       | a rule the adoption adds reaches both briefs                                       |
| C4       | two adopters from one parent leave one row; another child is outrun                |
| C5       | a snapshot whose contents do not match the manifest's digest is refused            |
| C6       | a critique citing a rule the firm holds is held                                    |
| C7       | two arms in hypothetical books of their own leave the canonical books untouched    |
"""

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault
from _firm import CAPPED, briefed, manifest, replayed, resolved
from _odoo import fresh_round

from examples.agent_firm.ar import Round
from examples.agent_firm.cases import SLA_CAP
from examples.agent_firm.knowledge import (
    ADOPTED,
    K0,
    K1,
    Adopted,
    Adopting,
    Knowledge,
    Mismatched,
    Outrun,
    Pinned,
    Rule,
    Unlinked,
    adopted,
    adoption_of,
)
from examples.agent_firm.manifest import head
from examples.agent_firm.saga import Completed

WRONG = Knowledge(rules=(Rule(id="POL-SLA-2", text="Credit whatever the customer deducted."),))
"""A rule the firm should not adopt."""


def _adopt(
    backend, adopting: Adopting, run_id: str | None = None, *, hypothetical: bool = False
) -> Any:
    run_id = run_id or f"a-{uuid4().hex[:8]}"

    def program(params: dict[str, Any]):
        return (yield from adopted(Adopting.admitted(params), params["run_id"]))

    backend.register(
        run_id, program, None, Fault(None), (), admitted=True, hypothetical=hypothetical
    )
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id, params=adopting.params()))
    assert snapshot.state == "completed", snapshot
    return snapshot.result


def _books() -> str:
    return f"k-{uuid4().hex[:8]}"


def test_c1_an_old_runs_replay_holds_after_the_head_advances(backend):
    books = _books()
    genesis = head(backend.ledger_read, books)
    run = resolved(backend, briefed(), manifest(genesis, Round(books)))
    assert Adopted.model_validate(_adopt(backend, Adopting(books=books, parent=genesis, child=K1)))
    assert head(backend.ledger_read, books) == Pinned(height=1, digest=K1.digest)
    assert replayed(backend, run) == run.result


def test_c2_a_no_op_delta_leaves_both_briefs_byte_identical(backend):
    books = _books()
    genesis = head(backend.ledger_read, books)
    _adopt(backend, Adopting(books=books, parent=genesis, child=K0))
    noop = head(backend.ledger_read, books)
    assert (noop.height, noop.digest) == (1, genesis.digest)
    before, after = briefed(), briefed()
    resolved(backend, before, manifest(genesis, fresh_round()))
    resolved(backend, after, manifest(noop, fresh_round()))
    assert (before.proposals, before.critiques) == (after.proposals, after.critiques)
    assert len(before.proposals) == len(before.critiques) == 1


def test_c3_a_rule_the_adoption_adds_reaches_both_briefs(backend):
    books = _books()
    _adopt(backend, Adopting(books=books, parent=head(backend.ledger_read, books), child=K1))
    firm = briefed()
    resolved(backend, firm, manifest(head(backend.ledger_read, books), Round(books)))
    ((proposal,), (critique,)) = firm.proposals, firm.critiques
    assert "POL-SLA-1" in proposal
    assert "POL-SLA-1" in critique


@pytest.mark.adversarial
def test_c4_two_adopters_from_one_parent_leave_one_row(backend):
    books = _books()
    genesis = head(backend.ledger_read, books)
    adopter = f"a-{uuid4().hex[:8]}"
    first = _adopt(backend, Adopting(books=books, parent=genesis, child=K1), adopter)
    rival = _adopt(backend, Adopting(books=books, parent=genesis, child=WRONG))
    again = _adopt(backend, Adopting(books=books, parent=genesis, child=K1))
    assert Adopted.model_validate(first) == Adopted.model_validate(again)
    assert Outrun.model_validate(rival) == Outrun(height=1, held=K1.digest)
    row = backend.ledger_read(adoption_of(books, 1))
    assert (row["kind"], row["child"], row["adopter"]) == (ADOPTED, K1.digest, adopter)
    assert backend.ledger_read(adoption_of(books, 2)) is None


def test_c5_a_snapshot_that_does_not_match_its_digest_is_refused(backend):
    edited = Knowledge(rules=(K1.rules[0].model_copy(update={"text": "Credit nothing."}),))
    firm = briefed(knowledge={K1.digest: edited})
    run = resolved(backend, firm, manifest(Pinned(height=1, digest=K1.digest), fresh_round()))
    assert run.snapshot.state == "failed", run.snapshot
    assert firm.proposals == []


@pytest.mark.parametrize(("knowledge", "memos"), [(K1, [Decimal(1000)]), (K0, [])])
def test_c6_a_critique_citing_a_rule_the_firm_holds_is_held(backend, knowledge, memos):
    books = _books()
    _adopt(
        backend, Adopting(books=books, parent=head(backend.ledger_read, books), child=knowledge)
    )
    firm = briefed(critique=CAPPED | {"cites": ["POL-SLA-1"]})
    run = resolved(backend, firm, manifest(head(backend.ledger_read, books), Round(books)))
    assert run.snapshot.state == "completed", run.snapshot
    erp = firm.world.erp
    posted = [
        entry.amount for entry in erp.entries(SLA_CAP.invoice.id) if entry.kind == "credit-memo"
    ]
    assert posted == memos


@pytest.mark.adversarial
def test_c7_two_hypothetical_arms_leave_the_canonical_books_untouched(backend):
    """Each arm settles in books of its own, under a hypothetical ledger; the canonical run then
    settles its own round at its own credit."""
    books = _books()
    genesis = head(backend.ledger_read, books)
    arms = [
        resolved(
            backend,
            briefed(),
            manifest(genesis, Round(f"{books}-{arm}"), critic=False),
            hypothetical=True,
        )
        for arm in ("a", "b")
    ]
    assert all(arm.snapshot.state == "completed" for arm in arms)
    assert [backend.ledger_payloads(arm.run_id) for arm in arms] == [[], []]
    canonical = resolved(backend, briefed(), manifest(genesis, Round(books)))
    assert "landed" in Completed.model_validate(canonical.snapshot.result).model_dump()


@pytest.mark.adversarial
def test_an_adoption_from_a_parent_off_the_chain_is_unlinked(backend):
    books = _books()
    genesis = head(backend.ledger_read, books)
    forged = Pinned(height=0, digest=WRONG.digest)
    assert Unlinked.model_validate(_adopt(backend, Adopting(books=books, parent=forged, child=K1)))
    _adopt(backend, Adopting(books=books, parent=genesis, child=K1))
    stale = Pinned(height=1, digest=WRONG.digest)
    assert Unlinked.model_validate(_adopt(backend, Adopting(books=books, parent=stale, child=K0)))
    assert head(backend.ledger_read, books) == Pinned(height=1, digest=K1.digest)


def test_the_head_refuses_a_height_that_does_not_adopt_from_the_one_below():
    rows = {adoption_of("b", 1): {"parent": WRONG.digest, "child": K1.digest}}
    with pytest.raises(Mismatched, match="height 1"):
        head(rows.get, "b")


@pytest.mark.adversarial
def test_a_hypothetical_adoption_leaves_the_canonical_head(backend):
    books = _books()
    genesis = head(backend.ledger_read, books)
    _adopt(backend, Adopting(books=books, parent=genesis, child=K1), hypothetical=True)
    assert head(backend.ledger_read, books) == genesis


@pytest.mark.adversarial
def test_a_canonical_adoption_of_a_height_a_hypothetical_one_holds_is_refused(backend):
    """The canonical books hold nothing at that height, so the adopter is not told it adopted."""
    books = _books()
    genesis = head(backend.ledger_read, books)
    _adopt(backend, Adopting(books=books, parent=genesis, child=K1), hypothetical=True)
    run_id = f"a-{uuid4().hex[:8]}"

    def program(params: dict[str, Any]):
        return (yield from adopted(Adopting.admitted(params), params["run_id"]))

    backend.register(run_id, program, None, Fault(None), (), admitted=True)
    adopting = Adopting(books=books, parent=genesis, child=K1)
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id, params=adopting.params()))
    assert (snapshot.state, backend.failure_kind(snapshot)) == ("failed", "LineageCollision")
    assert head(backend.ledger_read, books) == genesis
