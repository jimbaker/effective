"""Does a reader of the ledger survive the combinator that wrote it, and survive its schedule?

`test_shape_conformance.py` runs `fix` against `unfold` as a differential oracle: write a shape
twice and require the same answer, the same checkpoint names and the same ledger rows. This asks
that question one level out, of the READ side. A projection does not read rows; it folds them, and
a fold is a function of the order it receives them in unless something says otherwise.

Three readings of one run, in order of what they cost to believe:

| the reader | across spellings | across schedules |
|---|---|---|
| the rows themselves, as a set | agrees | agrees |
| a fold whose keys are disjoint | agrees | agrees |
| last write wins over one shared key | agrees | **disagrees** |

The spelling column belongs to the oracle and stays with it: `_shapes.agree` compares both the
rows and what they say across every spelling of every shape. The schedule column is this file.

The third row is the obligation, and it is pinned here as a fact rather than asserted away. The
ledger observable is a partial order (`docs/effective-design.md` §3.3); a run records one linear
extension of it, and a reader that is not invariant under the remaining extensions reads a run
rather than the record. Nothing in the substrate refuses such a reader, so the first thing to have
is a test that says which folds are safe.

The schedule is forced with `_schedules.Turnstile`, which is the op-layer seam of §3.6 turned
into an instrument. This lives outside `test_shape_conformance.py` and imports from it, because a
fold over its rows is a different subject from the shapes that write them.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from _schedules import Turnstile, ledger_path
from _shapes import Outcome, run
from test_shape_conformance import SPELLINGS, Budgeted, Tree, at

SHAPE = "halving"
FORKED = 1
"""The budget whose halving is one gather over two leaves: the smallest run with a schedule."""

type Rows = Sequence[Mapping[str, Any]]


# --- two readers of the same rows -----------------------------------------------------------


def by_entry(rows: Rows) -> dict[str, str]:
    """Key-disjoint: every leaf owns its own key, so no two rows land on one cell."""
    return {row["path"]: row["kind"] for row in rows}


def latest(rows: Rows) -> dict[str, str]:
    """Last write wins over `kind`, which every leaf row shares: the last row read takes the cell.

    The shape a real projection has whenever several events carry one entity's current state."""
    return {row["kind"]: row["path"] for row in rows}


SCHEDULES: dict[str, list[str]] = {"left first": ["0", "1"], "right first": ["1", "0"]}


def run_scheduled(backend, program: Budgeted, budget: int, order: Sequence[str]) -> Outcome:
    """`_shapes.run` with a turnstile holding the leaves' rows to `order`."""
    turnstile = Turnstile(order, ledger_path)
    outcome = run(backend, at(program, budget), Tree(), layers=lambda _run_id: [turnstile.layer()])
    assert turnstile.kept_its_schedule(), f"the run would not take {order}"
    return outcome


def read(backend, outcome: Outcome, reader: Callable[[Rows], dict[str, str]]) -> dict[str, str]:
    return reader(backend.ledger_payloads(outcome.run_id))


# --- the three readings -----------------------------------------------------------------------


@pytest.mark.parametrize("order", list(SCHEDULES.values()), ids=list(SCHEDULES))
@pytest.mark.parametrize("spelling", list(SPELLINGS[SHAPE]))
def test_a_key_disjoint_reader_is_a_function_of_the_record(backend, spelling, order):
    """Disjoint keys commute, so the fold reads the partial order rather than one extension."""
    outcome = run_scheduled(backend, SPELLINGS[SHAPE][spelling], FORKED, order)

    assert read(backend, outcome, by_entry) == {"0": "leaf", "1": "leaf"}


@pytest.mark.parametrize("spelling", list(SPELLINGS[SHAPE]))
def test_last_write_wins_over_a_shared_key_reads_the_schedule(backend, spelling):
    """The obligation, stated as the fact it is: this reader answers the run, not the record.

    Both schedules commit the same two rows and the gather joins in branch index order either
    way, so the workflow's own answer is unchanged. Only the fold moves, which is what makes it
    the reader's defect rather than the substrate's."""
    program = SPELLINGS[SHAPE][spelling]
    left = run_scheduled(backend, program, FORKED, SCHEDULES["left first"])
    right = run_scheduled(backend, program, FORKED, SCHEDULES["right first"])

    assert left.snap.result == right.snap.result == "01"
    assert read(backend, left, latest) == {"leaf": "1"}
    assert read(backend, right, latest) == {"leaf": "0"}
