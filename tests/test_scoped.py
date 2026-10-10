"""The structural scope `scoped(...)` on the recording/replay core.

A scope is an effect: `yield from scoped(atom, body)`, where the **handler** applies the prefix
in the same position and by the same mechanism as a gather's `gather:{g},{i};` branch
coordinate, so no author threads a `str` through callbacks and splices it onto every op name.
Three properties follow, and these tests pin them:

1. **The scope leaves the key.** Nothing in the workflow spells the prefix; the recorded key
   carries it, so hand-splicing cannot produce an adjacency collision.
2. **All three interpreters agree on the composed name**: recording, replay, and (in
   `test_conformance.py`) both durable engines. A key scheme that drifts between them is a
   silently orphaned checkpoint.
3. **A park inside a scope stays scoped.** In memory the parent's frame is held by a
   `ScopedSuspended`; durably it is re-derived by replay. Both must qualify the event name
   identically or an emitter cannot wake the run.

The scope atom is a `Key`, composed by the one composer, so its grammar is a property of
how the atom is built.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from effective.api import append_ledger, await_event, call_tool, gather, scoped
from effective.checkpoints import ENGINE_INTERNAL
from effective.cost import Usage
from effective.domain import DomainOp
from effective.handlers.base import op_key
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler, ScopedSuspended
from effective.handlers.replay import ReplayHandler
from effective.keys import Key, compose_key
from effective.keys.frame import ARM_TAGS
from effective.ops import CompositionRefused, LedgerRow, Scoped


def _keys(handler: RecordingHandler) -> list[str]:
    return [entry.key.stored() for entry in handler.trace]


class _SeqCtx:
    """A non-durable ctx that runs a step's thunk inline — enough to drive `DurableHandler`
    over the same workflows without an engine, so the third interpreter is covered here too."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def step(self, name: Key, thunk: Callable[[], Any]) -> Any:
        self.names.append(name.stored())
        return thunk()

    def await_event(self, name: Key) -> Any:
        raise NotImplementedError

    def sleep_until(self, when: Any, *, name: Key | None = None) -> None:
        raise NotImplementedError


class _EchoDomain:
    """Satisfies both `DomainInterpreter` and `MeteredDomain` — the fork drivers require the
    metered surface, and the guard under test fires before any domain call is made."""

    def run(self, op: Any) -> Any:
        return f"ran:{op.name}"

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        return self.run(op), Usage()


# --- 1. the scope leaves the key -------------------------------------------------------


def _leaf():
    return (yield from call_tool("a", {}, object))


def test_the_scope_is_applied_by_the_handler_not_written_by_the_author():
    """The workflow names `tool:a`; the recorded key is `rec:0;tool:a`. Nothing in the body
    mentions `rec:0` — that is the whole difference from the spliced form."""

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), _leaf))

    handler = RecordingHandler(responses={"tool:a": 1})
    assert handler.run(wf) == 1
    assert _keys(handler) == ["rec:0;step;tool:a"]


def test_two_scopes_over_one_body_do_not_alias():
    """The collision the hand-spliced scope existed to prevent, now prevented structurally:
    the same body under two atoms yields two distinct keys."""

    def wf():
        first = yield from scoped(compose_key(t"rec:{0}"), _leaf)
        second = yield from scoped(compose_key(t"rec:{1}"), _leaf)
        return (first, second)

    handler = RecordingHandler(responses={"tool:a": 1})
    handler.run(wf)
    assert _keys(handler) == ["rec:0;step;tool:a", "rec:1;step;tool:a"]


def test_nested_scopes_compose_the_prefix():
    def wf():
        return (
            yield from scoped(
                compose_key(t"rec:{1}"), lambda: scoped(compose_key(t"fold:{2},{0}"), _leaf)
            )
        )

    handler = RecordingHandler(responses={"tool:a": 1})
    handler.run(wf)
    assert _keys(handler) == ["rec:1;fold:2,0;step;tool:a"]


def test_the_scope_ends_with_the_body():
    """A scope is a dynamic extent, not a mode switch: the op after it is unscoped."""

    def wf():
        yield from scoped(compose_key(t"rec:{0}"), _leaf)
        return (yield from call_tool("a", {}, object))

    handler = RecordingHandler(responses={"tool:a": 1})
    handler.run(wf)
    assert _keys(handler) == ["rec:0;step;tool:a", "step;tool:a"]


def test_the_scope_reaches_every_arm_not_just_steps():
    """A ledger append and an artifact store are checkpointed too, so their keys are scoped
    as well — the prefix is applied to the op key, not to a step-name special case."""

    def body():
        yield from call_tool("a", {}, object)
        yield from append_ledger(LedgerRow(event_id=Key.parse("e1"), kind="k"))

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), body))

    handler = RecordingHandler(responses={"tool:a": 1})
    handler.run(wf)
    assert _keys(handler) == ["rec:0;step;tool:a", "rec:0;ledger;e1"]


# --- 2. the interpreters agree ---------------------------------------------------------


def test_replay_reproduces_the_recorded_scoped_keys():
    def wf():
        outer = yield from call_tool("outer", {}, object)
        inner = yield from scoped(compose_key(t"rec:{0}"), _leaf)
        return (outer, inner)

    handler = RecordingHandler(responses={"tool:outer": 1, "tool:a": 2})
    recorded = handler.run(wf)
    assert ReplayHandler(handler.trace).run(wf) == recorded


def test_the_durable_handler_composes_the_same_names():
    """The third interpreter. `_PrefixedCtx` is what applies the scope durably, and the ctx
    sees exactly the keys the recorder wrote — the two paths cannot drift to two schemes."""

    def wf():
        yield from call_tool("outer", {}, object)
        yield from scoped(compose_key(t"rec:{0}"), _leaf)

    ctx = _SeqCtx()
    DurableHandler(ctx, _EchoDomain()).run(wf)
    assert ctx.names == ["step;tool:outer", "rec:0;step;tool:a"]


def test_the_gather_ordinal_counts_across_a_scope_on_every_interpreter():
    """`gather:{g}:` numbering belongs to the handler, so a scope takes the next ordinal and the
    enclosing frame continues after it. A restart would let a scope entered twice name its gather
    the same both times. Pinned across all three interpreters: if they disagreed, a recorded branch
    key would stop matching the replayed or durable one."""

    def one(name: str):
        return lambda: call_tool(name, {}, object)

    def inner():
        return (yield from gather([one("g0")]))

    def wf():
        yield from gather([one("top")])
        yield from scoped(compose_key(t"rec:{0}"), inner)
        yield from gather([one("after")])

    expected = [
        "gather:0,0;step;tool:top",
        "rec:0;gather:1,0;step;tool:g0",
        "gather:2,0;step;tool:after",
    ]
    handler = RecordingHandler(responses={"tool:top": 1, "tool:g0": 2, "tool:after": 3})
    handler.run(wf)
    assert _keys(handler) == expected
    ReplayHandler(handler.trace).run(wf)  # the replay agrees or raises ReplayMismatch
    ctx = _SeqCtx()
    DurableHandler(ctx, _EchoDomain()).run(wf)
    assert ctx.names == expected


# --- 3. a park inside a scope stays scoped ---------------------------------------------


def _park_body():
    answer = yield from await_event("q", object)
    tail = yield from call_tool("after", {}, object)
    return (answer, tail)


def test_a_park_inside_a_scope_qualifies_the_event_name():
    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), _park_body))

    handler = RecordingHandler(responses={"tool:after": 9})
    parked = handler.run(wf)
    assert isinstance(parked, ScopedSuspended)
    assert parked.awaiting.stored() == "rec:0;q"  # the author wrote "q"
    assert parked.resume("ANSWER") == ("ANSWER", 9)
    assert _keys(handler) == ["rec:0;event;q", "rec:0;step;tool:after"]


def test_the_parent_resumes_after_the_scoped_body_returns():
    """The park propagates outward as the PARENT's park — on resume the body finishes and its
    value flows into the enclosing generator, which then runs on unscoped."""

    def wf():
        inner = yield from scoped(compose_key(t"rec:{0}"), _park_body)
        tail = yield from call_tool("tail", {}, object)
        return (inner, tail)

    handler = RecordingHandler(responses={"tool:after": 9, "tool:tail": 7})
    parked = handler.run(wf)
    assert parked.resume("A") == (("A", 9), 7)
    assert _keys(handler) == ["rec:0;event;q", "rec:0;step;tool:after", "step;tool:tail"]


def test_two_parks_in_one_scope_both_stay_scoped():
    def body():
        first = yield from await_event("q1", object)
        second = yield from await_event("q2", object)
        return (first, second)

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), body))

    handler = RecordingHandler()
    parked = handler.run(wf)
    assert parked.awaiting.stored() == "rec:0;q1"
    reparked = parked.resume("A1")
    assert isinstance(reparked, ScopedSuspended)
    assert reparked.awaiting.stored() == "rec:0;q2"
    assert reparked.resume("A2") == ("A1", "A2")
    assert _keys(handler) == ["rec:0;event;q1", "rec:0;event;q2"]


def test_a_scope_inside_a_gather_branch_qualifies_once_not_twice():
    """The double-prefix hazard: a `ScopedSuspended` surfacing as a gather branch's slot has
    ALREADY composed its name, so the gather must not prepend the branch path again."""

    def branch():
        return (yield from scoped(compose_key(t"s:{0}"), lambda: await_event("q", object)))

    def wf():
        return (yield from gather([lambda: call_tool("b0", {}, object), branch]))

    handler = RecordingHandler(responses={"tool:b0": 1})
    parked = handler.run(wf)
    assert parked.awaiting.stored() == "gather:0,1;s:0;q"
    assert parked.resume("A") == [1, "A"]


def test_resume_by_name_asserts_the_qualified_name():
    """The L2 by-name guard compares against the SCOPED name — the same string an engine would
    have delivered to, so a test that names the bare event fails here rather than in production."""

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), _park_body))

    handler = RecordingHandler(responses={"tool:after": 9})
    parked = handler.run(wf)
    with pytest.raises(ValueError, match="not 'q'"):
        parked.resume("ANSWER", name=Key.parse("q"))
    assert parked.resume("ANSWER", name=Key.parse("rec:0;q")) == ("ANSWER", 9)


# --- the refusals -----------------------------------------------------------------------


def test_a_scoped_has_no_standalone_op_key():
    """Pure structure, like a `Gather`: its effect is on the names of the ops inside it, so it
    has no key of its own and asking for one is a bug worth naming loudly."""
    with pytest.raises(ValueError, match="no standalone op key"):
        op_key(Scoped(scope=compose_key(t"rec:{0}"), body=_leaf))


def test_inspect_only_never_sees_a_scoped():
    """`inspect_only` answers "what may a counterfactual observe?"; a scope is structure, so the
    drivers interpret it themselves. Reaching here means a driver grew a `case _` that swallowed
    it, which would record the body's leaves under UNSCOPED keys — comparable to nothing the base
    run wrote. Refused loudly rather than fabricated."""
    from effective.sandbox import inspect_only

    with pytest.raises(TypeError, match="must never reach `inspect_only`"):
        inspect_only(Scoped(scope=compose_key(t"rec:{0}"), body=_leaf), {})


def test_a_scoped_body_returns_a_host_object_not_its_durable_form():
    """A scoped body's value flows back into a LIVE generator, so it must not be serialized on
    the way — only the task-result boundary does that.

    This is a regression, and the bug it pins was invisible to a workflow returning ints: the
    durable handler dumped a scoped body's return value to JSON, so `descend`'s `Deeper(...)`
    came back a dict, matched no arm of its decision table, and spun the trampoline forever."""

    @dataclass(frozen=True)
    class Verdict:
        value: int

    def body():
        yield from call_tool("a", {}, object)
        return Verdict(7)

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), body))

    ctx = _SeqCtx()
    assert DurableHandler(ctx, _EchoDomain()).run(wf) == {"value": 7}  # the TASK result IS dumped
    assert RecordingHandler(responses={"tool:a": 1}).run(wf) == Verdict(7)

    # …and inside the workflow the body's value is the object itself, never its dumped form.
    def uses_it():
        verdict = yield from scoped(compose_key(t"rec:{0}"), body)
        assert isinstance(verdict, Verdict), f"a scoped body returned {type(verdict).__name__}"
        return verdict.value

    assert DurableHandler(_SeqCtx(), _EchoDomain()).run(uses_it) == 7


def test_a_gather_branch_returns_a_host_object_too():
    """The same rule one composition over — `gather([scoped(...)])`, the flagship `recurse` shape.

    `_dump` belongs at the TASK boundary and nowhere else. A gather branch's value flows into the
    parent's live generator through the join, and branch results are never checkpointed (the join
    is reconstructed by replaying each branch), so serialising there was pure loss: the workflow
    saw a dataclass in memory and a `dict` durably. Regression for review finding F6, which is the
    scoped-body bug surviving in its sibling — the reason `_run`'s docstring now states the rule
    once instead of per call site."""

    @dataclass(frozen=True)
    class Verdict:
        value: int

    def body():
        yield from call_tool("a", {}, object)
        return Verdict(7)

    def wf():
        out = yield from gather([lambda: scoped(compose_key(t"rec:{0}"), body)])
        # asserted INSIDE the workflow: what the author's code binds is the whole question, and
        # the task result is dumped afterwards on either path, which would hide the divergence.
        assert isinstance(out[0], Verdict), f"branch handed the workflow a {type(out[0]).__name__}"
        return out[0].value

    ctx = _SeqCtx()
    assert DurableHandler(ctx, _EchoDomain()).run(wf) == 7
    assert RecordingHandler(responses={"tool:a": 1}).run(wf) == 7


def test_a_fork_point_cannot_be_a_scoped():
    """A fork point is a DECISION to substitute; a scope is structure with no key to match the
    `at` index on. Refused by CLASS, before the key check — `op_key` would raise there and mask
    the typed refusal (the same ordering `Gather` needs)."""
    from effective.fork import OpIndex, fork_at

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), _leaf))

    with pytest.raises(TypeError, match="cannot fork at a Scoped"):
        fork_at(wf, [], OpIndex(0), "substitute", _EchoDomain())


# --- 4. the two interpreters agree about LAYERS ------------------------------------------


def _observer(seen: list[str]):
    from effective.layers import op_layer

    @op_layer
    def observe(op):
        seen.append(type(op).__name__)
        result = yield op
        return result

    return observe


def test_no_op_layer_ever_sees_a_scoped():
    """A structural op has no `op_key` and no result of its own to authorize, so there is nothing
    for a layer to decide about it — the ops it CONTAINS are each layered individually.

    Pinned on both interpreters together because they disagreed: durable routed `Scoped` through
    the stack (`['Scoped','Step']`) and the recorder did not (`['Step']`), so a layer calling
    `op_key(op)` raised on one path only (review finding F3). Keyed on the shared
    `UNLAYERED_OPS` now, so the two cannot drift apart again."""

    def body():
        return (yield from call_tool("a", {}, object))

    def wf():
        return (yield from scoped(compose_key(t"rec:{0}"), body))

    recorded: list[str] = []
    RecordingHandler(responses={"tool:a": 1}, op_layers=(_observer(recorded),)).run(wf)

    durable: list[str] = []
    DurableHandler(_SeqCtx(), _EchoDomain(), op_layers=(_observer(durable),)).run(wf)

    assert recorded == durable == ["Step"]


def test_a_refused_escaping_a_scoped_body_reaches_the_parent():
    """`yield from scoped(a, body)` reads as inline code, so a body that declines to catch a
    refusal must hand the parent the same chance a plain `yield from` would.

    The durable handler and the recorder both deliver it. A gather branch keeps the opposite
    contract (branch failure is task-level), which is why `_run_structural` distinguishes them
    rather than treating 'structural' as one behavior."""
    from effective.govern import Refused
    from effective.layers import op_layer

    @op_layer
    def refuse(op):
        raise Refused(op, "nope")
        yield op  # pragma: no cover — never reached; the refusal is raised first

    def body():
        return (yield from call_tool("a", {}, object))

    def wf():
        try:
            return (yield from scoped(compose_key(t"rec:{0}"), body))
        except Refused:
            return "caught by the parent"

    assert RecordingHandler(responses={"tool:a": 1}, op_layers=(refuse,)).run(wf) == (
        "caught by the parent"
    )
    assert DurableHandler(_SeqCtx(), _EchoDomain(), op_layers=(refuse,)).run(wf) == (
        "caught by the parent"
    )


# --- a scope atom cannot forge an op's frame -----------------------------------------------


# Every head a scope atom may not open with: each op arm, and each marker the checkpoint readers
# set aside as engine bookkeeping, derived from `ENGINE_INTERNAL` so a new marker is covered.
FORGED_HEADS = [f"{arm}:1,0" for arm in ARM_TAGS] + [
    marker.strip(";") + "1" for marker in ENGINE_INTERNAL
]


@pytest.mark.parametrize("head", FORGED_HEADS)
def test_a_scope_atom_may_not_open_with_a_head_the_substrate_mints(head):
    """An atom tagged like an arm spells an op's own frame, as `gather:1,0` spells a branch the
    handler mints, so two different nestings could compose one path; an engine-internal head
    names bookkeeping the readers set aside, so the steps under it would vanish from them. Every
    walk's scope is a `Scoped` op, so refusing its construction reaches the recorder, the durable
    handler and replay alike, while the substrate still applies its own `gather:{g},{i}` frames."""
    forged = Key.parse(head)

    def wf():
        return (yield from scoped(forged, _leaf))

    with pytest.raises(CompositionRefused, match=r"op arm|engine bookkeeping"):
        Scoped(scope=forged, body=_leaf)
    with pytest.raises(ValueError, match=r"op arm|engine bookkeeping"):
        RecordingHandler(responses={"tool:a": 1}).run(wf)
    with pytest.raises(ValueError, match=r"op arm|engine bookkeeping"):
        DurableHandler(_SeqCtx(), _EchoDomain()).run(wf)
