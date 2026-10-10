"""Conformance: the LIVE edge decision vs the Lean model's rows.

The decision a task fails of, given a raised error and the attempt that raised it, is formalized
in `formal/lean/Effective/Retry.lean` as `failingLeaf`, and emitted to `formal/retry_vectors.json`
by `lake exe retry_vectors` (`just formal-vectors`). Each row is an attempt and a failure of one or
two leaves; this test builds the failure as an exception tree and runs the live `failing_leaf`.

| a leaf's property | built as                                           |
|-------------------|----------------------------------------------------|
| typed             | a subclass of `Unretryable`                        |
| rederived         | marked by `mark_rederived`                         |
| refusal           | a `Refused`                                        |
| undecided         | held inside a `RaceUndecided` group                |
"""

import json
from pathlib import Path
from typing import Any

import pytest

from effective.domain import CallTool
from effective.govern import Refused
from effective.handlers.base import Attempt, failing_leaf
from effective.ops import RaceUndecided, Step, Unretryable, mark_rederived

VECTORS: list[dict[str, Any]] = json.loads(
    (Path(__file__).resolve().parents[1] / "formal" / "retry_vectors.json").read_text()
)

pytestmark = pytest.mark.conformance


class _Raised(Exception):
    pass


class _Typed(_Raised, Unretryable):
    pass


class _TypedRefusal(Refused, Unretryable):
    pass


KINDS: dict[tuple[bool, bool], type[Exception]] = {
    (False, False): _Raised,
    (True, False): _Typed,
    (False, True): Refused,
    (True, True): _TypedRefusal,
}


OP = Step("t", CallTool(name="t", result_schema=dict))


def _leaf(spec: dict[str, bool], nested: bool) -> tuple[Exception, Exception]:
    """The leaf a spec describes, and what holds it in the failure: itself, or a plain group one
    level below, as a gather inside a race branch raises."""
    kind = KINDS[spec["typed"], spec["refusal"]]
    leaf = kind(OP, "refused") if issubclass(kind, Refused) else kind("raised")
    mark_rederived(leaf, spec["rederived"])
    inner: Exception = ExceptionGroup("a gather", [leaf]) if nested else leaf
    held = RaceUndecided("a race no branch can win", [inner]) if spec["undecided"] else inner
    return leaf, held


def _decided(row: dict[str, Any], nested: bool) -> int | None:
    built = [_leaf(spec, nested) for spec in row["leaves"]]
    leaves = [leaf for leaf, _ in built]
    held = [holder for _, holder in built]
    raised = held[0] if len(held) == 1 else ExceptionGroup("a failure", held)
    attempt = Attempt(number=3 if row["final"] else 1, limit=3, delayed=row["delayed"])
    fails_of = failing_leaf(raised, attempt)
    return (
        None if fails_of is None else next(i for i, leaf in enumerate(leaves) if leaf is fails_of)
    )


@pytest.mark.parametrize("nested", [False, True], ids=["a leaf", "a leaf in a group"])
def test_the_live_edge_decides_every_row_as_the_model_does(nested):
    disagree = [row for row in VECTORS if _decided(row, nested) != row["fails_of"]]
    assert disagree == []


def test_the_rows_reach_every_answer():
    """Each answer is in the table, so no row agrees by never asking."""
    assert {row["fails_of"] for row in VECTORS} == {None, 0, 1}
