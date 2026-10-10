"""An outer machine whose worker runs an inner one, and the one obligation that composition has.

ROLE: journey + adversarial. The happy arm proves two machines compose on a durable
store; the other proves the one way a composer gets it wrong is LOUD.

The composition itself needs no new machinery: `run_machine` is an ordinary effect-yielding
generator, so a worker may `yield from` another one and the frames nest. What it does need is a
rule about the inner address, because the postamble composes that from the `run_id`, and the
generations of a chain deliberately share one, so the run id cannot separate two machines alone.

**The rule: the inner run keeps the run id and says where it sits**, with `under`, and the
coordinates compose into the address as a sub-term. A run id names an execution; where a run sits
inside another is two coordinates, and packing them into the id leaves a kebab-formatted string
whose only legal spelling is an f-string in an identity position.
"""

import json
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import pytest

from effective.api import append_ledger, call_tool
from effective.coding.states import ReviewVerdict, State
from effective.coding.transition import transition
from effective.domain import DomainOp
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.handlers.durable import DurableHandler
from effective.keys import Name, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import build_specs
from effective.machine.trampoline import (
    Placement,
    Under,
    appending_states,
    canonical_violations,
    run_machine,
    running_under,
)
from effective.ops import LedgerRow

pytestmark = pytest.mark.journey

OUTER_RUN = "outer-1"

INNER_ROWS = [
    ("machine-committed", "under:review,0;machine:outer-1;commit"),
    ("machine-finished", "under:review,0;machine:outer-1"),
]
OUTER_ROWS = [
    ("machine-committed", "machine:outer-1;commit"),
    ("machine-finished", "machine:outer-1"),
]
"""The four addresses, spelled whole rather than built.

Two reasons, and only the first is about this file. A literal pins the BYTES, which is what an
assertion about a canonical address wants — rebuilding the expected value with the same composer
the trampoline uses would agree with itself through a rendering change. And an f-string in an
identity position is a finding by default here (`just key-check` said so about the first draft of
these four lines), which is the rule working exactly as intended on a test."""


class GreenSuite:
    """Every tool answers a passing predicate. The postamble runs on every exit path, so both
    machines yield one `run_suite` each and neither is what this file is measuring."""

    def run(self, op: DomainOp) -> Any:
        return CommandRun(exit_code=0)


def approving(_ctx: Ctx, _evidence: Evidence) -> Any:
    return ReviewVerdict.APPROVED
    yield  # pragma: no cover  -- a `Judge` is a generator


def unreachable(ctx: Ctx) -> Any:
    raise AssertionError(f"{ctx.state.value} should not run")
    yield  # pragma: no cover


def one_state_specs(worker, canonical: frozenset[State] = frozenset()):
    """REVIEW, approved, finish: the shortest walk that still reaches the postamble."""
    return build_specs(
        State,
        workers={State.REVIEW: worker},
        judges={State.REVIEW: approving},
        default_worker=unreachable,
        default_judge=approving,
        canonical=canonical,
    )


def inner_worker(ctx: Ctx) -> Any:
    """It YIELDS, and that is load-bearing rather than incidental.

    A worker that returns without yielding mints no op, so its state frame never reaches the tape
    at all — and every assertion about where the inner machine's keys land would then hold of an
    empty set. The first version of this file did exactly that and its nesting assertion failed
    honestly; had it been written as "no key lands outside the frames" it would have passed
    vacuously."""
    run: CommandRun = yield from call_tool("run_suite", {}, CommandRun)
    return Evidence(summary=f"the inner machine measured exit {run.exit_code}", measured=run)


def outer_worker_running(placing: Callable[[Ctx], Placement | None], inner=None):
    """A worker whose whole body is another machine. No new op, no new combinator.

    `placing` is a parameter so the obligation can be dropped and measured."""

    def worker(ctx: Ctx) -> Any:
        session = yield from run_machine(
            Run(OUTER_RUN),
            "inner goal",
            one_state_specs(inner or inner_worker),
            transition,
            start=State.REVIEW,
            budget=3,
            tree={"inner.py": "i"},
            under=placing(ctx),
        )
        announce(session)
        return Evidence(summary=f"inner path {[s.value for s in session.path]}")

    return worker


def sitting(ctx: Ctx) -> Placement:
    """Where the inner machine sits, READ OFF the `Ctx` the worker was handed.

    That is what makes the coordinates a fact rather than a claim: the state that ran the inner
    machine and the visit that ran it are both in hand here, so neither can be invented. Nothing
    outside the substrate spells the pair, which is why `running_under` takes the context."""
    return running_under(None, ctx)


def nowhere(ctx: Ctx) -> None:
    """The obligation dropped: an inner run that says nothing about where it sits."""
    return None


REPORTED: list[tuple[str, str]] = []
"""What each run says it appended, as `(commit, outcome)`, in the order the runs finish."""


def announce(session) -> None:
    REPORTED.append((session.commit_id.stored(), session.outcome_id.stored()))


@pytest.fixture
def app():
    REPORTED.clear()
    a = SqliteApp(":memory:")
    yield a
    a.close()


def drive(app, placing: Callable[[Ctx], Placement | None], canonical=frozenset(), inner=None):
    @app.register_task("nested")
    def task(params, ctx):
        rid = params["run_id"]

        def outer() -> Any:
            # Announced INSIDE the workflow: `DurableHandler.run` dumps at the task boundary, so
            # the caller is handed a `dict` and the run's account of itself has to be taken while
            # the `Session` is still one.
            session = yield from run_machine(
                Run(rid),
                "outer goal",
                one_state_specs(outer_worker_running(placing, inner), canonical),
                transition,
                start=State.REVIEW,
                budget=3,
                tree={"outer.py": "o"},
            )
            announce(session)
            return session

        return DurableHandler(
            ctx, GreenSuite(), ledger=SqliteLedger(app.conn, rid, app.write_lock)
        ).run(outer)

    return app.run_until_result(app.spawn("nested", {"run_id": OUTER_RUN}))


def ledger(app) -> list[tuple[str, str]]:
    return list(app.conn.execute("SELECT kind, event_id FROM ledger ORDER BY rowid"))


def test_two_machines_compose_and_both_reach_the_canonical_record(app):
    """The happy arm. The inner run keeps the run id and says where it sits, so the two addresses
    differ by a sub-term and all four rows land: a commit and an outcome per machine."""
    snap = drive(app, sitting)
    assert snap is not None
    assert snap.state == "completed", snap.failure

    assert ledger(app) == INNER_ROWS + OUTER_ROWS


def test_the_inner_machines_ops_land_INSIDE_the_outer_frames(app):
    """Frames nest, which is what makes the checkpoint identities disjoint even where the LEDGER
    addresses are not. That gap is the whole reason the test below has something to catch: scope
    frames separate checkpoints, and a canonical `event_id` is not a checkpoint."""
    drive(app, sitting)
    names = [n for (n,) in app.conn.execute("SELECT name FROM checkpoints")]
    nested = [n for n in names if n.count("state:") > 1]
    assert nested, f"no op ran under two state frames: {sorted(names)}"
    for name in nested:
        assert name.startswith("d:0;state:review;d:"), name


def test_an_inner_run_THAT_SAYS_NOTHING_is_REFUSED_rather_than_silently_dropped(app):
    """THE OBLIGATION, and the reason it can be an obligation rather than a hazard.

    Both machines compose their address from a `run_id`, so an inner run that adds no coordinates
    composes the same `event_id` twice. The store's ruling is total over (writer known?) x
    (holder known?) x (same task?) x (same placement?), and nesting lands on the one row that
    REFUSES: the inner machine
    runs inline, so it is the SAME task, while its ledger append sits under the outer's frames, so
    the placement differs.

    Measured rather than reasoned, on a real store, because the distinction is easy to get wrong:
    the rows "drop silently on the durable engine" for two runs in DIFFERENT tasks, and not for
    nesting.

    **Note WHICH rows survive**, because it is the inverse of what a reader expects: the inner
    postamble runs first, so the inner machine's rows are on the record and the OUTER run's are
    the ones lost. A composition that silently kept the sub-run and discarded the run is exactly
    the shape a loud refusal is worth having."""
    snap = drive(app, nowhere)
    assert snap is not None
    assert snap.state == "failed"
    assert "PlacedWriterCollision" in str(snap.failure or "")
    assert "machine:outer-1;commit" in str(snap.failure or "")

    # The inner machine's rows landed under the OUTER's address; the outer's own never did.
    assert ledger(app) == OUTER_ROWS


def payloads(app) -> list[dict[str, Any]]:
    rows = app.conn.execute("SELECT payload FROM ledger ORDER BY rowid").fetchall()
    return [json.loads(row[0]) for row in rows]


def test_an_outcome_row_carries_what_the_run_CONCLUDED(app):
    """The words, not only the measurement, so the address decodes to what a citation quotes.

    Without this the record holds a verdict, a path and a predicate's output, and a parent that
    composed its goal from a child's sentence can cite no row that contains the sentence."""
    drive(app, sitting)
    outcomes = [row for row in payloads(app) if row["kind"] == "machine-finished"]
    inner, outer = outcomes
    assert inner["summary"] == "the inner machine measured exit 0"
    assert outer["summary"] == "inner path ['review']"


def test_a_finished_run_NAMES_the_rows_it_appended(app):
    """A run that cannot name itself is a run nothing can cite, so a caller composed the skeleton
    a second time. The ids `Session` reports are the ones on the record."""
    drive(app, sitting)
    assert REPORTED == [
        ("under:review,0;machine:outer-1;commit", "under:review,0;machine:outer-1"),
        ("machine:outer-1;commit", "machine:outer-1"),
    ], REPORTED
    assert [event for _kind, event in ledger(app)] == [
        event for pair in REPORTED for event in pair
    ]


def test_the_state_that_RAN_the_inner_machine_reaches_the_canonical_record(app):
    """The audit's reading of a nesting, and the ruling it forces.

    The inner postamble appends inside the outer state's scope, so the tape says REVIEW reached
    the append-only ledger. It did: the machine REVIEW ran wrote those rows, and a nesting that
    the audit excused would be a state reaching the record with nothing to say so. A state that
    composes a machine declares `canonical`, and one that forgets is named rather than excused."""
    drive(app, sitting)
    names = [n for (n,) in app.conn.execute("SELECT name FROM checkpoints")]
    assert appending_states(State, names) == {
        State.REVIEW: [
            "d:0;state:review;ledger;under:review,0;machine:outer-1",
            "d:0;state:review;ledger;under:review,0;machine:outer-1;commit",
        ]
    }

    undeclared = one_state_specs(outer_worker_running(sitting))
    assert set(canonical_violations(State, undeclared, names)) == {State.REVIEW}
    declared = one_state_specs(outer_worker_running(sitting), frozenset({State.REVIEW}))
    assert canonical_violations(State, declared, names) == {}


def noting_inner_worker(ctx: Ctx) -> Any:
    """An inner worker that appends its OWN row, from inside the inner machine's state."""
    note = compose_key(t"note:{Name('inner')}")
    yield from append_ledger(LedgerRow(event_id=note, kind="noted"))
    return Evidence(summary="noted")


def test_a_row_minted_inside_the_inner_STATE_belongs_to_that_state_alone(app):
    """The other half of the nesting, and the one a per-term walk got wrong.

    This row's key carries a `state:` frame per machine it ran in. Attributing it to each reports
    one op under two states and counts it twice; the inner machine, placed under the outer
    REVIEW, owns it. Both frames read `review` here because both machines walk the same state
    type, so the placement is what the assertion turns on."""
    drive(app, sitting, inner=noting_inner_worker)
    names = [n for (n,) in app.conn.execute("SELECT name FROM checkpoints")]
    noted = [key for key in names if "note:inner" in key]
    assert noted == ["d:0;state:review;d:0;state:review;ledger;note:inner"], noted
    assert appending_states(State, noted) == {}
    assert appending_states(State, noted, under=Under(State.REVIEW.value, 0)) == {
        State.REVIEW: noted
    }


class Delegating(StrEnum):
    WORK = "work"


class Spending(StrEnum):
    SPEND = "spend"


def test_a_row_minted_under_another_enums_state_belongs_to_that_machine():
    """Two machines on two state types. The append is the inner machine's, so the outer machine's
    audit reports it under none of its own states."""
    noted = ["d:0;state:work;d:0;state:spend;ledger;note:inner"]
    assert appending_states(Delegating, noted) == {}
    inner = Under(Delegating.WORK.value, 0)
    assert appending_states(Spending, noted, under=inner) == {Spending.SPEND: noted}
