"""Op-key alignment: the fork's diff and the LID `~_H` projection, one core.

The central case: a fork pair's keys diverge at the FIRST LEDGER OP purely because the workflow
interpolates the run id into the event id (`ledger;r-base:e1` vs `ledger;r-fork-0:e1`). Without
run-id canonicalization a fork's divergence index is meaningless and `~_H` returns False for two
structurally identical runs, which is why `compare`/`equivalent` take a `scrub` argument;
canonicalizing gather/task scopes alone is not enough.

Two modules under test: `effective.lineage` (the alignment core and both projections) and
`effective.checkpoints` (the reader the projections consume: the `# --- readers` and
one-task-per-read sections below). Kept in one file because the reader cases exist to feed the
alignment cases, and separating them would put a fixture and its only assertion in two places.
"""

from uuid import UUID

import pytest

from effective.checkpoints import Checkpoint, from_trace, keys, read_sqlite_task
from effective.handlers.base import TraceEntry
from effective.keys import Key
from effective.lineage import RUN, Alignment, align, canonical, compare, equivalent, key_distance
from effective.ops import Step

# A ctx built directly rather than spawned, so its task id is a FIXTURE — a fixed literal
# keeps the checkpoint rows byte-stable. It read `"task-1"` until the id became a `UUID`;
# the banked store below is the one place a pre-uuid7 TEXT id still has to be readable.
_TASK = UUID("019fa000-0000-7000-8000-000000000001")

BASE = ("tool:fetch", "extract", "ledger;r-base:e1", "artifact:app/json:abc")
FORK = ("tool:fetch", "extract", "ledger;r-fork-0:e1", "artifact:app/json:abc")


# --- the alignment core -------------------------------------------------------------------


def test_identical_sequences_share_everything_and_never_diverge():
    a = align(BASE, BASE)
    assert a.identical
    assert not a.diverged
    assert a.shared_prefix == len(BASE)
    assert a.first_divergence is None


def test_a_prefix_is_not_a_divergence():
    """One run truncated (a crash, a park) is a shorter run, not a different one."""
    a = align(BASE, BASE[:2])
    assert not a.diverged
    assert a.first_divergence is None
    assert a.shared_prefix == 2


def test_divergence_is_reported_at_the_first_differing_index():
    a = align(BASE, FORK)
    assert a.diverged
    assert a.first_divergence == 2
    assert a.shared_prefix == 2


def test_empty_sequences_align_trivially():
    a = align((), ())
    assert a.identical
    assert not a.diverged
    assert a.shared_prefix == 0


# --- canonicalization: the run id in an event id ----------------------------------------


def test_WITHOUT_scrubbing_a_fork_pair_diverges_at_its_first_ledger_op():
    """The motivating case: an unscrubbed run id reads as a divergence, at the ledger op."""
    assert not equivalent(BASE, FORK)


def test_WITH_run_id_scrubbing_the_same_pair_is_structurally_identical():
    assert equivalent(BASE, FORK, scrub=("r-base", "r-fork-0"))
    assert compare(BASE, FORK, scrub=("r-base", "r-fork-0")).shared_prefix == len(BASE)


def test_scrubbing_replaces_the_token_wherever_it_appears():
    assert canonical(("govern:review:r-9:0", "budget-grant:r-9,2"), scrub=("r-9",)) == (
        f"govern:review:{RUN}:0",
        f"budget-grant:{RUN},2",
    )


def test_a_longer_token_is_scrubbed_before_a_token_that_prefixes_it():
    """`r-1` must not half-scrub `r-10`, which would silently equate two different runs."""
    assert canonical(("ledger;r-10:e1",), scrub=("r-1", "r-10")) == (f"ledger;{RUN}:e1",)


def test_a_mapping_keeps_distinct_scopes_distinct():
    """A caller that wants run and task scopes told apart says so."""
    assert canonical(("t-7;ledger;r-2:e1",), scrub={"r-2": RUN, "t-7": "{task}"}) == (
        "{task};ledger;{run}:e1",
    )


def test_scrubbing_is_the_callers_call_not_a_guess():
    """Nothing is scrubbed by default. A guesser that stripped a MEANINGFUL substring would
    equate runs that genuinely differ, so the default is to change nothing."""
    assert canonical(BASE) == BASE


# --- projection 1: the keys distance -------------------------------------------------------


def test_distance_is_zero_exactly_when_the_runs_are_equivalent():
    assert key_distance(align(BASE, BASE)) == 0
    assert key_distance(compare(BASE, FORK, scrub=("r-base", "r-fork-0"))) == 0
    assert key_distance(align(BASE, FORK)) > 0


def test_distance_counts_keys_outside_the_matching_blocks():
    a = ("x", "y", "z")
    assert key_distance(align(a, ("x", "y", "z"))) == 0
    assert key_distance(align(a, ("x", "q", "z"))) == 2  # one substitution: one out, one in
    assert key_distance(align(a, ("x", "y"))) == 1  # one deletion
    assert key_distance(align(a, ("x", "y", "z", "w"))) == 1  # one insertion


def test_distance_is_symmetric():
    for other in (FORK, BASE[:2], (*BASE, "extra")):
        assert key_distance(align(BASE, other)) == key_distance(align(other, BASE))


# --- readers -------------------------------------------------------------------------------


def test_a_trace_reads_as_the_same_shape():
    trace = [
        TraceEntry(Key.parse("tool:fetch"), Step(name="tool:fetch", op=None), "e"),  # ty: ignore[invalid-argument-type]
        TraceEntry(Key.parse("extract"), Step(name="extract", op=None), "r"),  # ty: ignore[invalid-argument-type]
    ]
    assert keys(from_trace(trace)) == ("tool:fetch", "extract")


def test_checkpoint_state_is_carried_raw():
    """The fork's event transplant needs the payload; decoding here would strip what the v1
    meter folds."""
    envelope = {"result": "ok", "usage": {"cost": 0.01}}
    assert Checkpoint(Key.parse("m1"), envelope).state == envelope


def test_alignment_is_a_frozen_value():
    a = align(BASE, FORK)
    assert isinstance(a, Alignment)
    with pytest.raises(AttributeError):
        a.shared_prefix = 99  # ty: ignore[invalid-assignment]


# --- projection 2: the ledger-lineage marginal ---------------------------------------------


def _row(eid: str, **kw):
    return {"event_id": eid, **kw}


def test_marginal_splits_two_lineages_at_the_divergence():
    """The fork's deliverable: 'approve -> commits $42; reject -> ledgers a rejection'. Both
    lineages extract and review (shared structure); they diverge at the CONSEQUENCE (base commits,
    fork ledgers a rejection), which is different event kinds."""
    from effective.lineage import marginal

    base = [_row("extracted:r-base"), _row("reviewed:r-base"), _row("committed:r-base", amount=42)]
    fork = [_row("extracted:r-fork"), _row("reviewed:r-fork"), _row("rejected:r-fork")]
    m = marginal(base, fork, scrub=("r-base", "r-fork"))

    assert m.shared_prefix == 2  # extracted + reviewed align once run-id is scrubbed
    assert m.diverged
    assert m.base_tail == ({"event_id": "committed:r-base", "amount": 42},)  # ORIGINAL, unscrubbed
    assert m.fork_tail == ({"event_id": "rejected:r-fork"},)


def test_marginal_aligns_by_IDENTITY_so_a_substituted_payload_shows_downstream():
    """The contract worth stating: `marginal` aligns by event IDENTITY (the `key`), not payload. A
    decision fork substitutes a value AT an event that keeps its id (`reviewed:*` in both), so that
    event ALIGNS and the divergence surfaces at its consequence, not at the substitution itself.
    When the fork point is recorded (the genesis' `forked_at_event`), a caller slices
    there directly; this is the discovery form, for arbitrary runs."""
    from effective.lineage import marginal

    base = [_row("reviewed:r-a", decision="approve"), _row("committed:r-a")]
    fork = [_row("reviewed:r-b", decision="reject")]
    m = marginal(base, fork, scrub=("r-a", "r-b"))
    assert m.shared_prefix == 1  # the reviewed event aligns despite the differing decision payload
    assert m.base_tail == ({"event_id": "committed:r-a"},)  # the divergence is the consequence
    assert m.fork_tail == ()


def test_marginal_without_scrub_diverges_at_the_first_run_scoped_event():
    """Same lesson as `compare`: unscrubbed, two structurally identical lineages share nothing,
    because every event id embeds the run id."""
    from effective.lineage import marginal

    base = [_row("extracted:r-base"), _row("reviewed:r-base")]
    fork = [_row("extracted:r-fork"), _row("reviewed:r-fork")]
    assert marginal(base, fork).shared_prefix == 0
    assert marginal(base, fork, scrub=("r-base", "r-fork")).shared_prefix == 2


def test_two_identical_lineages_have_no_marginal():
    from effective.lineage import marginal

    same = [_row("a:r"), _row("b:r")]
    m = marginal(same, same, scrub=("r",))
    assert not m.diverged
    assert m.base_tail == ()
    assert m.fork_tail == ()


def test_marginal_keys_on_event_id_by_default_and_accepts_a_custom_key():
    """Default identity is `event_id`; a caller can align on `kind` instead (a coarser diff)."""
    from effective.lineage import marginal

    base = [_row("e1", kind="extract"), _row("e2", kind="review")]
    fork = [_row("e9", kind="extract"), _row("e8", kind="commit")]
    m = marginal(base, fork, key=lambda r: r["kind"])
    assert m.shared_prefix == 1  # both `extract` first
    assert [r["kind"] for r in m.fork_tail] == ["commit"]


def test_read_sqlite_task_returns_decoded_python_values(tmp_path, sqlite_app):
    """Parity with Absurd's jsonb reader and `from_trace`: `Checkpoint.state` is a Python VALUE,
    not the raw JSON string — so the stage-3 seed reader can re-commit it through `ctx.step`
    without double-encoding, as Absurd's reader does."""
    from effective.sqlite import SqliteTaskContext

    db = tmp_path / "task.db"
    app = sqlite_app(str(db))
    envelope = {"result": {"amount": "4.50"}, "usage": {"cost": 0.01}}
    SqliteTaskContext(app.conn, _TASK, app.write_lock).step(Key.parse("extract"), lambda: envelope)

    recorded = read_sqlite_task(str(db), _TASK)
    assert [c.key.stored() for c in recorded] == ["extract"]
    assert recorded[0].state == envelope  # the whole {result, usage} envelope...
    assert isinstance(recorded[0].state, dict)  # ...as a Python dict, NOT a JSON string


# --- one task per read ---------------------------------------------------------------------
#
# `Checkpoint` carries no task id, so a read that returned EVERY task's checkpoints interleaved in
# rowid order would let `fork_seed`, which returns at the FIRST match for its `through` key,
# silently describe whichever lineage came first. A parallel marginal sweep (N children off ONE
# base in ONE store) makes lineage mixing the steady state.
#
# `task_id` is required, so these pin the CLASS (a mixed read is unrepresentable) rather than one
# collision.


def _two_lineages_in_one_store(tmp_path, sqlite_app):
    """One store, two tasks, the SAME step name in each: the shape a mixed read makes ambiguous."""
    from effective.api import ask_llm
    from effective.cost import Usage
    from effective.handlers.absurd import DurableHandler

    class _Tagged:
        def __init__(self, tag: int) -> None:
            self.tag = tag

        def run(self, op):
            return {"from": self.tag}

        def run_metered(self, op):
            return self.run(op), Usage()

    def wf():
        value = yield from ask_llm("extract", [], dict)
        return value

    db = tmp_path / "two.db"
    app = sqlite_app(str(db))
    ids = {}
    for run_id, tag in (("r-one", 11), ("r-two", 22)):

        @app.register_task(f"b-{run_id}")
        def base_task(params, ctx, _tag=tag):
            return DurableHandler(ctx, _Tagged(_tag), ledger=None).run(wf)

        ids[run_id] = app.spawn(f"b-{run_id}", {})
        app.run_until_result(ids[run_id])
    return str(db), ids


def test_a_read_is_scoped_to_one_task(tmp_path, sqlite_app):
    """Each lineage's `extract` checkpoint carries ITS OWN value — no interleaving."""
    db, ids = _two_lineages_in_one_store(tmp_path, sqlite_app)
    one = read_sqlite_task(db, ids["r-one"])
    two = read_sqlite_task(db, ids["r-two"])
    assert [c.state for c in one] == [{"from": 11}]
    assert [c.state for c in two] == [{"from": 22}]


def test_a_mixed_lineage_read_is_unrepresentable(tmp_path, sqlite_app):
    """The fix, as a type-level property: there is no way to ask for "every task"."""
    db, _ids = _two_lineages_in_one_store(tmp_path, sqlite_app)
    with pytest.raises(TypeError, match="task_id"):
        read_sqlite_task(db)  # ty: ignore[missing-argument]


def test_fork_seed_over_a_task_scoped_read_takes_the_intended_lineage(tmp_path, sqlite_app):
    """The product: seeding a fork off the SECOND base gets the second base's value, which is what
    silently failed before — `fork_seed` returns at the first `through` match, and both lineages
    used the same step name."""
    from effective.fork import fork_seed

    db, ids = _two_lineages_in_one_store(tmp_path, sqlite_app)
    seed = fork_seed(read_sqlite_task(db, ids["r-two"]), through="step:extract")
    assert seed[Key.parse("step:extract")] == {"from": 22}
