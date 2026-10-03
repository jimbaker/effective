"""`scripts/vacuity_probe.py`'s own pins — each naming the mutation that reddens it.

The instrument shipped with two defects a green run could not show, both found by reading its
OUTPUT rather than its code: it watched two of the store's four key-bearing seams, and it
perturbed `peek_event` without perturbing `emit_event`, which desynchronised delivery and
manufactured two findings. These pin the repairs.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "vacuity_probe", Path(__file__).resolve().parent.parent / "scripts" / "vacuity_probe.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
vacuity_probe = importlib.util.module_from_spec(_SPEC)
sys.modules["vacuity_probe"] = vacuity_probe
_SPEC.loader.exec_module(vacuity_probe)

from effective.keys import Segment, compose_key  # noqa: E402


def rows(*triples: tuple[str, str]) -> list[dict]:
    return [{"test": t, "key": "k", "occurrence": 1, "event": e} for t, e in triples]


# ---------------------------------------------------------------- the shift


def test_the_shift_is_textually_identical_across_a_Key_seam_and_a_str_seam():
    """Reddens if `SHIFT` stops being a plain scope prefix.

    The store is reached through both `Key`-typed seams (`step`, `peek_event`) and `str`-typed
    ones (`emit_event`, `idempotency_key`). If the two spellings diverged, a perturbed run would
    fail for the instrument's reason instead of the test's — which is exactly what happened when
    `emit_event` was left unperturbed."""
    key = compose_key(t"review:{Segment('m1')}")
    assert key.prefixed(vacuity_probe.SHIFT).stored() == vacuity_probe.SHIFT + key.stored()


def test_the_shift_lands_inside_the_language():
    """Reddens if `SHIFT` stops being a key a real scope could have minted.

    A perturbation that produced an unparseable name would redden every grammar assertion in the
    suite and report them all as spelling-observed."""
    from effective.keys.grammar import parse

    parse(compose_key(t"review:{Segment('m1')}").prefixed(vacuity_probe.SHIFT).stored())


def test_the_shift_is_idempotent_because_the_seams_call_each_other():
    """Reddens if the shift stops being a projection.

    `await_event` delegates to `self.peek_event`, which is the patched wrapper, so a naive
    shift applies twice and stores `probe:0;probe:0;…` — a name no reader and no resume can
    match. Seams calling each other is the normal case, not a corner."""
    shift = vacuity_probe._shifter(True)
    once = shift(compose_key(t"review:{Segment('m1')}"))
    assert vacuity_probe._text(shift(once)) == vacuity_probe._text(once)


def test_the_shift_is_off_when_not_perturbing():
    """Reddens if the baseline pass ever perturbs. Its outcomes are the control."""
    key = compose_key(t"review:{Segment('m1')}")
    assert vacuity_probe._shifter(False)(key) is key


def test_the_shift_is_injective_so_distinctness_survives():
    """Reddens if the perturbation ever collapses two keys onto one.

    Distinctness preservation is the whole reason a perturbed run still resolves its checkpoints;
    a collapsing shift would answer a different question (and break every durable test)."""
    a = compose_key(t"review:{Segment('m1')}").prefixed(vacuity_probe.SHIFT).stored()
    b = compose_key(t"review:{Segment('m2')}").prefixed(vacuity_probe.SHIFT).stored()
    assert a != b


# ---------------------------------------------------------------- text coercion


def test_a_Key_at_a_str_annotated_seam_is_unwrapped_not_repred():
    """Reddens if `_text` falls back to `str()` on a `Key`.

    `SqliteTaskContext.emit_event` annotates `name: str` and is handed a `Key`, so the recorder
    meets both. `str(key)` is the REPR — recording that would put `Key(_value=…)` in the report
    and silently split one key into two rows."""
    key = compose_key(t"review:{Segment('m1')}")
    assert vacuity_probe._text(key) == "review:m1"
    assert vacuity_probe._text("review:m1") == "review:m1"


# ---------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        (("write", "hit"), "resolving"),
        (("await", "event_hit"), "resolving"),
        (("await", "miss"), "no-store"),
        (("write", "await"), "write-only"),
        (("write", "write"), "write-only"),
        (("miss",), "no-store"),
        (("write", "emit"), "unjudged-seam"),
        (("write", "idempotency"), "unjudged-seam"),
        (("write", "emit", "hit"), "resolving"),
    ],
)
def test_classify_is_total_over_the_event_alphabet(events, expected):
    """Reddens if a new event kind falls through to the wrong bucket.

    `emit` and `idempotency` must NOT reach `write-only`: they are selective seams the injective
    perturbation cannot judge, and calling them inert would claim a domain this tool does not
    scan. That misclassification shipped, and a test named
    `…never_alias_two_idempotency_keys_onto_one_task` is what exposed it."""
    assert vacuity_probe.classify(rows(*(("t", e) for e in events)))["t"] == expected


# ---------------------------------------------------------------- the verdict join


@pytest.mark.parametrize(
    ("cls", "observed", "expected"),
    [
        ("resolving", False, "resolving"),
        ("resolving", True, "resolving+observed"),
        ("write-only", True, "spelling-observed"),
        ("write-only", False, "key-inert"),
        ("unjudged-seam", False, "unjudged-seam"),
        ("unjudged-seam", True, "spelling-observed"),
        ("no-store", False, "key-inert"),
    ],
)
def test_the_verdict_join_is_total(cls, observed, expected):
    """Reddens if the two-pass join loses an arm.

    The join is the finding: a key did work if the store CONSULTED it or the test OBSERVED its
    spelling. Every (class, observed) pair needs a written answer, not a fall-through."""
    assert vacuity_probe._label(cls, observed=observed) == expected


def test_a_test_that_fails_unperturbed_is_never_judged():
    """Reddens if the verdict scores an already-red test.

    A test failing in the baseline pass tells us nothing about its keys, and counting it as
    key-inert would let an unrelated breakage inflate the finding list."""
    out = vacuity_probe.verdict(
        rows(("f.py::t", "write")),
        perturbed={"f.py::t": "failed"},
        baseline={"f.py::t": "failed"},
    )
    assert "f.py::t" not in out


def test_key_inert_is_reported_as_measured_not_as_a_verdict_of_worthlessness():
    """Reddens if the bucket is renamed back to a conclusion.

    It was called `VACUOUS` for one afternoon. The perturbation is property-preserving, so a
    test asserting a PROPERTY of the key language passes under it while doing real work —
    `test_no_durable_name_anywhere_carries_a_Key_REPR` is the worked case. The label states what
    was measured; a reader supplies the judgement."""
    out = vacuity_probe.verdict(
        rows(("f.py::t", "write")), perturbed={"f.py::t": "passed"}, baseline={"f.py::t": "passed"}
    )
    assert "key-inert" in out
    assert "VACUOUS" not in out
