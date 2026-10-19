"""An event name is an identity: composed as a `Key`, or one tag as text.

The subjects `("x,y", "z")` and `("x", "y,z")` would serialize to one name, `ev:x,y,z`, whichever
coordinate holds the comma. So every door that takes a name as text refuses text holding key
structure, and `Segment` refuses the coordinate that would carry it into a composed name."""

import pickle
from datetime import UTC, datetime

import pytest

from effective import coroutine_api
from effective.api import GatherBranch, await_event, await_until, qualified_event_name
from effective.keys import Key, Segment, compose_key
from effective.keys.grammar import METACHARACTER_ROLES, METACHARACTERS
from effective.ops import (
    FormattedEventName,
    OpIdentityAwaited,
    ReservedEventName,
    emitted_name,
    event_name,
)

_FORMATTED = "ev:x,y,z"


def _first_op(effect):
    """The op an effect yields first, which is where its name is checked."""
    return next(effect)


_DOORS = [
    ("event_name", event_name),
    ("emitted_name", emitted_name),
    ("await_event", lambda n: _first_op(await_event(n, dict))),
    ("await_until", lambda n: _first_op(await_until(n, dict, deadline=datetime.now(UTC)))),
    ("coroutine_api.await_event", lambda n: coroutine_api.await_event(n, dict)),
    ("qualified_event_name", lambda n: qualified_event_name(GatherBranch(0, 1), name=n)),
]
_IDS = [door for door, _ in _DOORS]


@pytest.mark.parametrize(("door", "take"), _DOORS, ids=_IDS)
def test_every_door_refuses_a_formatted_name(door, take):
    with pytest.raises(FormattedEventName):
        take(_FORMATTED)


def test_a_door_refuses_text_holding_any_metacharacter_and_names_its_role():
    """The refusal is over the grammar's whole set, so a new metacharacter is refused here too."""
    named = {}
    for byte in METACHARACTERS:
        with pytest.raises(FormattedEventName) as refused:
            event_name(f"ev{byte}x")
        named[byte] = f"the {METACHARACTER_ROLES[byte]} {byte!r}" in str(refused.value)
    assert named == dict.fromkeys(METACHARACTERS, True)


@pytest.mark.parametrize(("door", "take"), _DOORS, ids=_IDS)
def test_every_door_takes_one_tag_and_a_composed_key(door, take):
    """Anti-vacuity for the refusal: a door refusing every name would pass it."""
    take("go")
    take(compose_key(t"ev:{Segment('x')},{Segment('z')}"))


@pytest.mark.parametrize("subjects", [("x,y", "z"), ("x", "y,z")], ids=["left", "right"])
def test_a_composed_name_refuses_the_coordinate_that_would_collide(subjects):
    left, right = subjects
    with pytest.raises(ValueError, match="arity separator"):
        compose_key(t"ev:{Segment(left)},{Segment(right)}")


def test_each_engines_emitter_refuses_a_formatted_name(backend):
    with pytest.raises(FormattedEventName):
        backend.emit_event(None, _FORMATTED, {})


def test_each_engines_emitter_takes_a_structured_key(backend):
    """A stored name is structured, and `Key.parse` is where its reader makes it a key again."""
    backend.emit_event(None, Key.parse("gather:0,1;ev:x"), {})


def test_the_qualifier_takes_text_through_the_await_door_and_a_key_as_it_stands():
    """An emitter wakes substrate parks too, so a parsed authority key passes the qualifier."""
    with pytest.raises(ReservedEventName):
        qualified_event_name(GatherBranch(0, 1), name="approve")
    parked = Key.parse("depth-grant:r1,0")
    with pytest.raises(ReservedEventName):
        event_name(parked)
    assert qualified_event_name(GatherBranch(0, 1), name=parked).stored() == (
        "gather:0,1;depth-grant:r1,0"
    )


@pytest.mark.parametrize(
    "refusal",
    [
        FormattedEventName("ev:x"),
        ReservedEventName("approve", parsed=True),
        OpIdentityAwaited("x"),
    ],
    ids=lambda refusal: type(refusal).__name__,
)
def test_each_refusal_rebuilds_from_its_arguments(refusal):
    """Each refusal rebuilds from its arguments, as `pickle` and `copy` rebuild an exception."""
    rebuilt = pickle.loads(pickle.dumps(refusal))
    assert (type(rebuilt), str(rebuilt)) == (type(refusal), str(refusal))
