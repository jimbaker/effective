"""A relative await's name is one term, and no unscoped name carries an authority tag."""

from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault

from effective.api import await_event, qualified_event_name
from effective.budget import depth_grant_name
from effective.keys import Key, Name, compose_key

GRANT = depth_grant_name("r1", depth=1, generation=0)


@pytest.mark.parametrize(
    ("awaited", "refusal"),
    [
        (Key.parse("rec:0;depth-grant:r1,0,depth=1"), "ReservedEventName"),
        (Key.parse("foo:0;bar:x"), "FramedEventName"),
        (qualified_event_name(compose_key(t"lane:{Name('p')}"), name=GRANT), "FramedEventName"),
    ],
    ids=["authority tag under a frame", "framed name", "framed settlement name"],
)
def test_an_unscoped_framed_await_fails_its_run_on_the_engines(backend, awaited, refusal):
    def awaits(run_id: str):
        return (yield from await_event(awaited, dict))

    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(run_id, awaits, CountingDomain(), Fault(None), ())
    snapshot = backend.run_until_result(backend.spawn(run_id, run_id))
    assert snapshot is not None
    assert (snapshot.state, snapshot.failure.kind) == ("failed", refusal)
