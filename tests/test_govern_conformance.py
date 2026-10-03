"""Conformance: the LIVE `govern.combine` vs the Lean model's machine-derived rows.

The third instance of the method (`test_enforce_measured_conformance.py` — the measured trip;
`test_decide_conformance.py` — the permission cascade): formalize the transition once, emit
`decide`-verified rows, run the live Python against them. A fourth gate — a quota, a rate limit —
is enrolled by adding a policy, not by re-proving the ruling.

Beyond the rows, two properties from `Govern.lean` are re-checked against the live function over
every permutation of every row, because they are the ones an implementation gets backwards:
`combine_kind_perm_invariant` (a council has no argument order) and `refuse_dominates`.

`(A-tags)`: the model tags asks and reasons with abstract Nats; Python carries `Ask` objects and
reason strings. The mapping is positional and total — tag `n` becomes `Ask("p{n}", ...)` /
`"reason-{n}"` — so a row's fused payload ORDER is observable, which is the half of the fold that
is not order-free.
"""

import json
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest

from effective.govern import Ask, Park, Proceed, Refuse, combine

_VECTORS_PATH = Path(__file__).resolve().parents[1] / "formal" / "govern_vectors.json"
VECTORS: list[dict[str, Any]] = json.loads(_VECTORS_PATH.read_text())


# One model, one law, many arms: the overlap across arms IS the design, so the unit-role
# minimize-overlap rule does not apply here.
pytestmark = pytest.mark.conformance


def _ask(tag: int) -> Ask:
    return Ask(f"p{tag}", f"ask-{tag}")


def _verdict(spec: dict[str, Any]):
    match spec:
        case {"kind": "proceed"}:
            return Proceed()
        case {"kind": "park", "asks": asks}:
            return Park(tuple(_ask(t) for t in asks))
        case {"kind": "refuse", "reasons": reasons}:
            return Refuse(tuple(f"reason-{t}" for t in reasons))
    raise AssertionError(f"unknown verdict in the vector file: {spec}")


def _row_id(row: dict[str, Any]) -> str:
    return "+".join(v["kind"][:3] for v in row["verdicts"]) or "empty"


ROWS = [pytest.param(row, id=_row_id(row)) for row in VECTORS]


@pytest.mark.parametrize("row", ROWS)
def test_the_ruling_matches_the_model(row: dict[str, Any]):
    """Constructor AND fused payload, in argument order."""
    assert combine([_verdict(v) for v in row["verdicts"]]) == _verdict(row["ruling"])


@pytest.mark.parametrize("row", ROWS)
def test_every_permutation_yields_the_same_KIND(row: dict[str, Any]):
    """`combine_kind_perm_invariant`, checked on the live function: a council of peers has no
    argument order. (Contrast `serve`, whose parameter order IS its semantics.)"""
    verdicts = [_verdict(v) for v in row["verdicts"]]
    kinds = {type(combine(list(order))) for order in permutations(verdicts)}
    assert kinds == {type(combine(verdicts))}


@pytest.mark.parametrize("row", ROWS)
def test_refuse_dominates_and_no_objection_is_silently_dropped(row: dict[str, Any]):
    """`refuse_dominates` + `no_silent_proceed` on the live function."""
    verdicts = [_verdict(v) for v in row["verdicts"]]
    ruling = combine(verdicts)
    if any(isinstance(v, Refuse) for v in verdicts):
        assert isinstance(ruling, Refuse)
    if any(not isinstance(v, Proceed) for v in verdicts):
        assert not isinstance(ruling, Proceed)
