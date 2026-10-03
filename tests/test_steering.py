"""`SteeringCtx` on both engines: a principal's answer, substituted at a live op.

The point of parametrizing these over `backend` rather than pinning them on SQLite is the rule
that **a green SQLite pass is not a port.** The substitution here
is claimed to be engine-agnostic *because* `TaskContext` is the interface both engines
implement, and a claim about the seam is only worth what the second engine says about it.

Four properties, in the order they would break:

1. the workflow sees the principal's value, and the domain is never asked;
2. the CHECKPOINT holds it, which is what makes a steered run replay to the same place;
3. a steer whose coordinate missed is reported rather than swallowed;
4. occurrences steer independently: `#2` without `#1`.

(4) is the adversarial one, and its two failure modes are not symmetric.
Keying on the bare name steers BOTH occurrences, and its `first` assertion catches that on both
engines. Passing the occurrence-resolved key DOWN to the inner ctx is caught on **SQLite only**,
and measuring why is worth more than the count.

Under that mutation SQLite fails: `Key.occurrence` refuses a key that already carries a
`#N` suffix, so the task dies before writing anything. Absurd COMPLETES, and its two checkpoints
come back under exactly `TOOL_A` and `TOOL_A.occurrence(2)`, the names the unmutated run
assigns. The same rule arrives twice: both `Key.occurrence` and the vendored SDK's
duplicate-checkpoint rule omit the suffix at count 1, so re-suffixing an already-suffixed key is
a no-op there. The mutation is harmless on Absurd and refused on SQLite, and that asymmetry is a
fact about the seams: the typed `Key` boundary carries a guard that the path converting to `str`
at the SDK edge cannot.

The checkpoint assertion in (4) stays regardless. It pins that a steered value lands at the key the
ENGINE assigns, which is what `replay(tape) = tape` rests on: a value at a key no replay binds to
is a steer that did not happen.
"""

from uuid import uuid4

from _conformance import CountingDomain, Fault

from effective.api import call_tool
from effective.keys import Key, Segment
from effective.steering import Steer, SteeringCtx

REVIEWER = Segment("reviewer")
"""The principal these cases attribute to — a `Segment`, so it is a well-formed key atom."""

TOOL_A = Key.parse("step;tool:a")
TOOL_B = Key.parse("step;tool:b")


def _two_tools(_rid: str):
    a = yield from call_tool("a", {}, int)
    b = yield from call_tool("b", {}, int)
    return {"a": a, "b": b}


def _repeated_tool(_rid: str):
    first = yield from call_tool("a", {}, int)
    second = yield from call_tool("a", {}, int)  # the SAME step name, twice
    return {"first": first, "second": second}


def _run_steered(backend, workflow, steers, *, domain=None):
    """Register `workflow` under a `SteeringCtx` carrying `steers`, run it, return everything a
    case asserts on.

    `made` is a list because the task body runs once per attempt, so a resumed or retried task
    builds a fresh ctx; the last one is the attempt that finished. Capturing them all keeps that
    visible instead of asserting against whichever instance happened to be first.
    """
    domain = domain if domain is not None else CountingDomain()
    run_id = f"steer-{uuid4().hex[:8]}"
    name = f"steer-{run_id}"
    made: list[SteeringCtx] = []

    def wrap(ctx):
        made.append(SteeringCtx(ctx, steers))
        return made[-1]

    backend.register(name, workflow, domain, Fault(), (), wrap=wrap)
    task_id = backend.spawn(name, run_id)
    snap = backend.run_until_result(task_id)
    assert snap is not None
    assert snap.state == "completed", snap
    return snap, domain, made[-1], task_id


def test_a_steered_step_answers_from_the_principal_and_never_calls_the_domain(backend):
    """The core claim. `CountingDomain` answers `b` with 20, so a result of 99 is the principal's
    value and could have come from nowhere else — and `calls` is what proves the domain was not
    merely overruled after the fact but never asked."""
    snap, domain, ctx, _ = _run_steered(
        backend, _two_tools, {TOOL_B: Steer(value=99, by=REVIEWER)}
    )

    assert snap.result == {"a": 10, "b": 99}
    assert domain.calls == ["a"]
    assert ctx.applied == {TOOL_B: REVIEWER}
    assert ctx.unapplied() == frozenset()


def test_the_checkpoint_holds_the_steered_value(backend):
    """What makes a steered run durable rather than a one-off: the substitution commits by the
    ordinary `ctx.step` path, so the record a replay binds to carries the principal's value. If
    this fails while the case above passes, the steer reached the workflow and missed the
    store, and the run would replay to a DIFFERENT place: `replay(tape) = tape` holds iff the
    steer is in the tape."""
    _, _, _, task_id = _run_steered(backend, _two_tools, {TOOL_B: Steer(value=99, by=REVIEWER)})

    states = backend.checkpoint_states(task_id)
    assert states[TOOL_B] == 99
    assert states[TOOL_A] == 10  # the unsteered neighbour is untouched


def test_a_steer_the_run_never_reaches_is_reported_unapplied(backend):
    """A coordinate that missed — a key from a tape the workflow no longer produces. The run must
    complete normally (a steer is not a demand), and the miss must be legible afterwards, because
    silence here is indistinguishable from a steer that fired."""
    stale = Key.parse("step;tool:zzz")
    snap, domain, ctx, _ = _run_steered(backend, _two_tools, {stale: Steer(value=1, by=REVIEWER)})

    assert snap.result == {"a": 10, "b": 20}  # everything ran live
    assert domain.calls == ["a", "b"]
    assert ctx.applied == {}
    assert ctx.unapplied() == frozenset({stale})


def test_occurrences_steer_independently(backend):
    """`#2` steered, `#1` live. `CountingDomain(incrementing=True)` returns a DISTINCT value per
    call, so `first == 11` says the first occurrence really reached the domain rather than
    happening to match a stale checkpoint."""
    second = TOOL_A.occurrence(2)
    snap, domain, ctx, task_id = _run_steered(
        backend,
        _repeated_tool,
        {second: Steer(value=77, by=REVIEWER)},
        domain=CountingDomain(incrementing=True),
    )

    assert snap.result == {"first": 11, "second": 77}
    assert domain.calls == ["a"]  # only the first occurrence was asked
    assert ctx.applied == {second: REVIEWER}
    assert ctx.unapplied() == frozenset()

    # At the record, not just at the workflow — and on BOTH engines, which the assertions above
    # do not manage (see the module docstring's note on how the two differ in loudness). A
    # steered value at a key the engine did not assign is a value no replay binds to.
    states = backend.checkpoint_states(task_id)
    assert states[TOOL_A] == 11
    assert states[second] == 77
