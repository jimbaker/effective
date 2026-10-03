"""The permission cascade's decision as a PURE transition (governed-boundaries §8.2).

`decide` is to permission what `enforce_measured` is to budget: one total, deterministic,
fail-closed definition of the ruling, lifted out of the driver that realizes it. These are the
pure-axis pins; `tests/test_decide_conformance.py` holds the live function to the Lean model's
machine-derived rows, and `tests/test_permission.py` covers the driver (`cascade`) end to end.

The load-bearing property here is **suffix absorption**: a decisive verdict absorbs everything
after it, which is exactly what makes the driver's eager short-circuit sound — and the
short-circuit is not an optimization, it is the reason a settled cascade never parks a human it
does not need.
"""

from itertools import product

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
    decisive,
    rules,
)

OP = AppendLedgerRow(row=LedgerRow(event_id=Key.parse("e1"), kind="commitment"))

# The verdict alphabet, with distinguishable reasons so "which one won" is observable.
ALPHABET: tuple[Verdict, ...] = (Allow(), Deny("blocked"), Escalate("defer"))


def _sequences(length: int) -> list[tuple[Verdict, ...]]:
    return list(product(ALPHABET, repeat=length))


def test_decide_is_total_over_every_short_sequence():
    """Always a `Decision` — never an `Escalate`, never a `None` fall-through."""
    for length in range(4):
        for seq in _sequences(length):
            assert isinstance(decide(seq), Allow | Deny)


def test_decide_is_deterministic():
    """A pure function of (verdicts, default) — same inputs, same ruling, every time."""
    for seq in _sequences(3):
        first = decide(seq)
        assert decide(seq) == first
        assert decide(iter(seq)) == first  # and independent of the iterable's flavor


def test_first_decisive_verdict_wins():
    for seq in _sequences(3):
        settled = [v for v in seq if decisive(v)]
        expected: Verdict = settled[0] if settled else FAIL_CLOSED
        assert decide(seq) == expected


def test_all_escalate_falls_to_the_default_fail_closed():
    for length in range(4):
        assert decide([Escalate()] * length) == FAIL_CLOSED
        assert isinstance(decide([Escalate()] * length), Deny)


def test_an_escalate_default_is_a_misconfiguration_resolved_fail_closed():
    """A `default` that cannot settle is not a fall-through hole — it denies."""
    assert decide([], default=Escalate()) == MISCONFIGURED_DEFAULT
    assert decide([Escalate()], default=Escalate()) == MISCONFIGURED_DEFAULT


def test_an_explicit_allow_default_is_honored():
    """The fold does not hardcode fail-closed — `default` is the caller's law (the
    governed-boundaries §10 question: park-vs-deny as the SHARED default is `govern`'s
    call, not something baked into permission's fold)."""
    assert decide([], default=Allow()) == Allow()
    assert decide([Escalate(), Escalate()], default=Allow()) == Allow()


def test_decisive_partitions_the_alphabet():
    assert [decisive(v) for v in ALPHABET] == [True, True, False]


def test_a_decisive_verdict_absorbs_every_suffix():
    """**Suffix absorption** — the property that makes the driver's short-circuit sound.

    If a prefix ends decisive, appending ANY continuation cannot change the ruling. So the
    driver, which stops collecting at the first decisive verdict, computes the same decision
    the full tier list would have — while never running the tiers that would have parked."""
    for prefix in _sequences(2):
        if not decisive(prefix[-1]):
            continue
        for suffix in _sequences(2):
            assert decide(prefix + suffix) == decide(prefix)


@pytest.mark.parametrize("seq", _sequences(3))
def test_the_driver_realizes_exactly_what_decide_rules(seq: tuple[Verdict, ...]):
    """Driver/transition agreement: `cascade` over rules-tiers emitting `seq` forwards iff
    `decide(seq)` allows, and raises `Refused` with the ruling's reason iff it denies."""
    gate = cascade([rules(lambda op, v=v: v) for v in seq])
    match decide(seq):
        case Allow():
            assert drive_through([gate], OP, lambda op: "forwarded") == "forwarded"
        case Deny(reason=reason):
            with pytest.raises(Refused) as ei:
                drive_through([gate], OP, lambda op: "forwarded")
            assert ei.value.reason == reason
