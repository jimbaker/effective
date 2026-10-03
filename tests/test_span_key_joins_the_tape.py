"""The right-orphan assertion: every span's address is on the tape, on both engines.

ROLE: conformance. The telemetry join has a fixed DIRECTIONALITY, and this gate holds it.
Telemetry is the third bookkeeper (the tape re-serves, the ledger is canonical, spans observe),
so no bookkeeper may be derived from another, and the join is a **left outer join from the
tape**. A node with no span is
normal (telemetry off, sampled out, the recording core publishes no placement). A span whose
address is NOT on the tape is a defect, so the right-orphan set asserts EMPTY.

What that one assertion buys: it catches the entire class "someone added a second address minter",
which is otherwise silent. A second minter produces plausible rows and a plausible row count, and
the loss shows up only if you count.

The workflow is chosen to be the defect. Two calls to the SAME tool with no `AskLLM` between them
share one `iteration`, so `span_id`, which is `(session, name, iteration)`, collides by
construction. On that shape a spanId-keyed backend keeps 2 of 3 rows. The placed key separates
them, so this file pins the collision AND its repair in one run.
"""

import contextlib
from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault

from effective.api import call_tool, scoped
from effective.keys import compose_key
from effective.layers import compose_domain
from effective.permission import Allow, Deny, Refused, cascade, rules
from effective.telemetry import Span, genai_attributes, traced

pytestmark = pytest.mark.conformance

SCOPE = "probe"


def colliding_and_scoped_wf(run_id: str):
    """Two calls to ONE tool (the `span_id` collision), then one under a scope (the frames)."""
    first = yield from call_tool("a", {}, int)
    second = yield from call_tool("a", {}, int)
    third = yield from scoped(compose_key(t"probe:1"), lambda: call_tool("b", {}, int))
    return {"first": first, "second": second, "third": third}


def test_no_span_addresses_a_node_the_tape_does_not_have(backend):
    spans: list[Span] = []
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"orphan-{run_id}"
    # COMPOSED INTO THE DOMAIN. `register(..., layers=)` installs OP layers, whose alphabet is
    # `Step` — `traced` is a DOMAIN layer and a probe at the wrong seam proves nothing about it.
    domain = compose_domain((traced(spans.append, session_id=run_id),), CountingDomain())
    backend.register(name, colliding_and_scoped_wf, domain, Fault(None), ())
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "completed", snap

    tape = backend.checkpoint_keys(task_id)
    assert tape == ["step;tool:a", "step;tool:a#2", "probe:1;step;tool:b"], tape

    # NOT VACUOUS. A right-orphan set is empty over no spans and over all-`None` keys too, so the
    # assertion below means nothing until these two hold. The arc that built this file is the one
    # that learned a green instrument can be green over the empty set.
    assert len(spans) == 3, [s.name for s in spans]
    assert all(s.key is not None for s in spans), [(s.name, s.key) for s in spans]

    addressed = {s.key.stored() for s in spans if s.key is not None}
    assert addressed - set(tape) == set(), addressed - set(tape)

    # Two calls to `a` in one turn: their placed keys differ by the occurrence the walk minted,
    # and the span id is derived from the placed key, so the ids differ too.
    a_spans = [s for s in spans if s.tool_name == "a"]
    assert len({s.span_id for s in a_spans}) == 2, "two calls must not share a span id"
    assert {s.key.stored() for s in a_spans if s.key is not None} == {
        "step;tool:a",
        "step;tool:a#2",
    }

    # PLACED, not local: the scoped op's address carries its frame. A local key would render every
    # one of these `tool:{name}` and address nothing.
    scoped_span = next(s for s in spans if s.tool_name == "b")
    assert scoped_span.key is not None
    assert scoped_span.key.stored().startswith(f"{SCOPE}:1;")

    # Flattened ONCE, at the sink boundary — and it is the STORED form, which is what replay binds
    # to. `display()` here would be a second spelling of the identity.
    attrs = genai_attributes(scoped_span)
    assert attrs["effective.key"] == scoped_span.key.stored()


def deny_first_then_allow():
    """The shipped cascade, refusing exactly once."""
    asked: list[int] = []

    def policy(op):
        asked.append(1)
        return Deny("first call refused") if len(asked) == 1 else Allow()

    return cascade([rules(policy)])


def refused_then_retried_wf(run_id: str):
    """Ask, be refused, route around the refusal, ask again."""
    with contextlib.suppress(Refused):
        yield from call_tool("a", {}, int)
    return {"second": (yield from call_tool("a", {}, int))}


@pytest.mark.xfail(
    strict=True,
    reason="`_place` counts every op routed through the op-layer stack; the engines count "
    "only ops reaching `ctx.step`. A layer that short-circuits desynchronizes them permanently "
    "for that name; the fix is to unify the placement minters.",
)
def test_no_span_addresses_a_node_the_tape_lacks_when_a_layer_SHORT_CIRCUITS(backend):
    """The same assertion as above, over the composition the file above never walks.

    **This file's docstring claims the empty right-orphan set "catches the entire class *someone
    added a second address minter*".** There are already two, and they agree only on the path the
    happy-path workflow takes. `AbsurdHandler._place` is evaluated BEFORE `drive_through(...)`, so
    it burns an occurrence for an op the layer stack refuses; `permission.cascade`'s `Deny` arm
    raises `Refused` without ever forwarding, so `ctx.step`, where the engines keep their own
    counter, is never called. The retry then commits as `step;tool:a` while its span says
    `step;tool:a#2`.

    **Pinned rather than fixed.** Joining `_place`'s counter against a second bookkeeper, the
    span, turns a disagreement that only ever had to agree with itself into a wrong number on a
    dashboard; the fix is to unify the placement minters.

    Pinned with the SHIPPED cascade rather than a bespoke short-circuiting layer, because the
    question a reader will ask is whether this reaches production, and the answer is that this is
    the layer the live drain installs."""
    spans: list[Span] = []
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"shortcircuit-{run_id}"
    domain = compose_domain((traced(spans.append, session_id=run_id),), CountingDomain())
    backend.register(
        name, refused_then_retried_wf, domain, Fault(None), (deny_first_then_allow(),)
    )
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)

    assert snap is not None
    assert snap.state == "completed", snap
    tape = backend.checkpoint_keys(task_id)
    # ANTI-VACUITY, the same two the case above carries: the assertion below is empty over no
    # spans and over all-`None` keys, and this file's own arc is where that lesson was learned.
    assert len(spans) == 1, [s.name for s in spans]
    assert all(s.key is not None for s in spans), [(s.name, s.key) for s in spans]

    addressed = {s.key.stored() for s in spans if s.key is not None}
    assert addressed - set(tape) == set(), (
        f"a span addresses a node the tape does not have: {addressed - set(tape)}; tape={tape}"
    )
