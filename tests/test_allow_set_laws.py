"""The allow-set's laws, over every key a small grammar generates.

`Key.covers` is a partial order, and a `PermitPolicy` holds the entries no other entry covers,
sorted. So a policy depends on what its entries permit and on nothing about how they were
written: any order, any duplicate, any redundant entry gives the same value.
"""

from itertools import combinations, permutations, product

import pytest

from effective.keys import Key, Segment, compose_key
from effective.permission import PermitPolicy

pytestmark = pytest.mark.property

_COORDINATES = ((), ("a",), ("b",), ("a", "b"))
_NAMED = (("n=a",), ("a", "n=b"))


_A = Segment("a")


def _framed(key: Key) -> Key:
    """`key` under a `frame:a` frame: an identity where it opens with an arm, else an address."""
    match key.terms()[0].tag:
        case "step":
            # lint: terminal-hole: `key` is a `Key`.
            return compose_key(t"frame:{_A};{key:domain=identity}")
        case _:
            # lint: terminal-hole: `key` is a `Key`.
            return compose_key(t"frame:{_A};{key:domain=address}")


def _keys(coordinates: tuple[tuple[str, ...], ...]) -> list[Key]:
    terms = [
        tag + (":" + ",".join(coords) if coords else "")
        for tag, coords in product(("step", "tool"), coordinates)
    ]
    unframed = [Key.parse(term) for term in terms]
    framed = [_framed(key) for key in unframed]
    return unframed + framed + [key.occurrence(2) for key in unframed + framed]


KEYS = _keys(_COORDINATES)
# The order is read over named coordinates too, where a name-blind comparison would let `step:a`
# and `step:n=a` cover each other; the allow-set laws stay on the smaller domain for time.
ORDERED_KEYS = _keys(_COORDINATES + _NAMED)
ALLOW_SETS = [entries for size in range(4) for entries in combinations(KEYS, size)]


def _permitted(policy: PermitPolicy) -> frozenset[Key]:
    return frozenset(key for key in KEYS if policy.permits(key))


def _covered(entries: tuple[Key, ...]) -> frozenset[Key]:
    return frozenset(key for key in KEYS if any(entry.covers(key) for entry in entries))


def test_covers_is_a_partial_order():
    offenders = [
        (a, b, c)
        for a, b, c in product(ORDERED_KEYS, repeat=3)
        if not a.covers(a)
        or (a.covers(b) and b.covers(a) and a != b)
        or (a.covers(b) and b.covers(c) and not a.covers(c))
    ]
    assert offenders == []


def test_a_policy_permits_exactly_what_its_written_entries_cover():
    offenders = [s for s in ALLOW_SETS if _permitted(PermitPolicy(allow=s)) != _covered(s)]
    assert offenders == []


def test_every_order_of_the_entries_is_one_policy():
    offenders = [
        s
        for s in ALLOW_SETS
        if len({PermitPolicy(allow=order).allow for order in permutations(s)}) != 1
    ]
    assert offenders == []


def test_allow_sets_that_permit_the_same_keys_are_one_policy():
    by_coverage: dict[frozenset[Key], set[tuple[Key, ...]]] = {}
    for s in ALLOW_SETS:
        by_coverage.setdefault(_covered(s), set()).add(PermitPolicy(allow=s).allow)
    assert [policies for policies in by_coverage.values() if len(policies) != 1] == []


def test_the_canonical_form_is_a_fixed_point_and_no_entry_covers_another():
    offenders = [
        policy
        for policy in (PermitPolicy(allow=s) for s in ALLOW_SETS)
        if PermitPolicy(allow=policy.allow) != policy
        or any(a != b and a.covers(b) for a in policy.allow for b in policy.allow)
        or list(policy.allow) != sorted(policy.allow, key=Key.stored)
    ]
    assert offenders == []
