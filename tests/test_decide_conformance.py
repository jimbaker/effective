"""Conformance: the LIVE permission cascade vs the Lean model's machine-derived rows.

The cascade's decision is formalized once in `formal/lean/Effective/Decide.lean`. Its
`conformanceVectors` are `decide`-verified against the Lean transition (`conformance_vectors_hold`,
axiom-free) and emitted to `formal/decide_vectors.json` by `lake exe decide_vectors`
(`just formal-vectors`). This test runs the live Python against every row, at both levels:

  - **the transition** — `permission.decide`, the pure fold;
  - **the driver** — `permission.cascade`, which realizes the ruling (forward the op vs raise
    `Refused`) and, separately, must run tiers only up to the first decisive verdict.

This is the sibling of `tests/test_enforce_measured_conformance.py` (the measured trip), and it
closes the asymmetry the governed-boundaries note named: permission now has the same
machine-derived reference budget already had.

`(A-reason)`: the Lean model tags a `deny` reason with an abstract Nat; Python carries strings.
The mapping is `REASONS` below — tags 0/1 are the two module-level constants (so a change to
either reason string fails here), and tags >= 2 are tier-supplied reasons whose only requirement
is that they travel with the winning verdict. That reals-to-tags boundary is discharged by
sampling, not proved in Lean — the same posture `EnforceMeasured.lean` takes for dollars.
"""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from effective.keys import Key
from effective.layers import drive_through
from effective.ops import AppendLedgerRow, LedgerRow
from effective.permission import (
    FAIL_CLOSED,
    MISCONFIGURED_DEFAULT,
    Allow,
    Deny,
    Escalate,
    Refused,
    Verdict,
    cascade,
    decide,
    rules,
)

_VECTORS_PATH = Path(__file__).resolve().parents[1] / "formal" / "decide_vectors.json"
VECTORS: list[dict[str, Any]] = json.loads(_VECTORS_PATH.read_text())

OP = AppendLedgerRow(row=LedgerRow(event_id=Key.parse("e1"), kind="commitment"))

# The model's abstract reason tags -> the live reason strings.
REASONS = {0: FAIL_CLOSED.reason, 1: MISCONFIGURED_DEFAULT.reason}


# One model, one law, many arms: the overlap across arms IS the design, so the unit-role
# minimize-overlap rule does not apply here.
pytestmark = pytest.mark.conformance


def _reason(tag: int) -> str:
    return REASONS.get(tag, f"tier-{tag}")


def _verdict(spec: dict[str, Any]) -> Verdict:
    match spec["kind"]:
        case "allow":
            return Allow()
        case "deny":
            return Deny(_reason(spec["reason"]))
        case _:
            return Escalate()


def _row_id(row: dict[str, Any]) -> str:
    kinds = "+".join(v["kind"][:3] for v in row["verdicts"]) or "empty"
    return f"{kinds}|default={row['default']['kind'][:3]}"


ROWS = [pytest.param(row, id=_row_id(row)) for row in VECTORS]


def test_the_vector_file_is_the_lean_projection():
    """A guard on the pipeline itself: the committed file must be the emitter's JSON, not a
    hand-edited table (and not `lake build` progress output — that bug was live in the recipe)."""
    assert len(VECTORS) >= 15
    assert all({"verdicts", "default", "decision"} == set(row) for row in VECTORS)


@pytest.mark.parametrize("row", ROWS)
def test_the_transition_matches_the_model(row: dict[str, Any]):
    """`permission.decide` — the pure fold — agrees with the Lean model on every row."""
    verdicts = [_verdict(v) for v in row["verdicts"]]
    ruling = decide(verdicts, _verdict(row["default"]))
    match row["decision"]:
        case {"kind": "allow"}:
            assert ruling == Allow()
        case {"kind": "deny", "reason": tag}:
            assert ruling == Deny(_reason(tag))


@pytest.mark.parametrize("row", ROWS)
def test_the_driver_realizes_the_model_ruling(row: dict[str, Any]):
    """`cascade` — the driver — forwards the op exactly when the model allows, and raises
    `Refused` carrying the model's winning reason exactly when it denies."""
    verdicts = [_verdict(spec) for spec in row["verdicts"]]
    tiers = [rules(lambda op, v=v: v) for v in verdicts]
    gate = cascade(tiers, default=_verdict(row["default"]))
    match row["decision"]:
        case {"kind": "allow"}:
            assert drive_through([gate], OP, lambda op: "forwarded") == "forwarded"
        case {"kind": "deny", "reason": tag}:
            with pytest.raises(Refused) as ei:
                drive_through([gate], OP, lambda op: "forwarded")
            assert ei.value.reason == _reason(tag)
            assert ei.value.op is OP


@pytest.mark.parametrize("row", ROWS)
def test_the_driver_stops_at_the_first_decisive_tier(row: dict[str, Any]):
    """Suffix absorption, observed on the live driver: tiers after the first decisive verdict
    never RUN. The model licenses skipping them (`suffix_absorbed`); this pins that the driver
    actually does — the saving is not cycles, it is not parking a human needlessly."""
    ran: list[int] = []

    def _tier(i: int, v: Verdict):
        def policy(op, i=i, v=v):
            ran.append(i)
            return v

        return rules(policy)

    verdicts = [_verdict(spec) for spec in row["verdicts"]]
    gate = cascade([_tier(i, v) for i, v in enumerate(verdicts)], default=_verdict(row["default"]))
    with suppress(Refused):
        drive_through([gate], OP, lambda op: "forwarded")

    decisive_at = next((i for i, v in enumerate(verdicts) if not isinstance(v, Escalate)), None)
    expected = list(range(len(verdicts) if decisive_at is None else decisive_at + 1))
    assert ran == expected
