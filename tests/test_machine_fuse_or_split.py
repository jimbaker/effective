"""A judgement can be FUSED into its worker's visit or be a STATE of its own, and both know where
they are.

ROLE: conformance. A walk that hardcodes a two-phase sub-machine (run a worker, then a judge,
always exactly that shape) bakes a graph in as a product: `StateSpec` carries two slots and only
one of them gets a `Ctx`. This file reproduces what that asymmetry costs.

The embodiment lives in `tests/_conformance.py`, so that the durable conformance lane can drive
the same walk on both engines under crash injection. The three cases that do sit in
`test_conformance.py`, under the ruling-machine heading.

What stays here is a SQLite driver over that shared embodiment, and it earns its place two ways.
It SPELLS the ids it asserts, where the durable lane composes them, so it is the half of the pair
that can see a change to the minter. And it holds the `Ctx`-hashability and `Report.of`-parity
pins, which are properties of the types rather than of any engine.

Driven on the embedded SQLite engine rather than the recorder, deliberately. `RecordingHandler`
mints DUPLICATE keys where both durable engines apply the SDK's `name#N` rule, so a recorder tape
cannot answer a question about key or id uniqueness, which is the whole question here.
"""

from collections.abc import Callable
from dataclasses import fields

import pytest
from _conformance import (
    RulingDeployment,
    RulingState,
    WorkVerdict,
    ruling_machine_wf,
)

from effective.handlers.absurd import DurableHandler
from effective.keys import Segment
from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence, Report
from effective.sqlite import SqliteApp, SqliteLedger

pytestmark = pytest.mark.conformance


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


@pytest.fixture
def walk(app):
    deployment = RulingDeployment()

    @app.register_task("fs")
    def task(params, ctx):
        rid = params["run_id"]
        return DurableHandler(
            ctx, deployment, ledger=SqliteLedger(app.conn, rid, app.write_lock)
        ).run(lambda: ruling_machine_wf(rid))

    snap = app.run_until_result(app.spawn("fs", {"run_id": "fs-1"}))
    return snap, deployment


def ledger_ids(app) -> list[str]:
    return [e for (e,) in app.conn.execute("SELECT event_id FROM ledger ORDER BY rowid")]


def keys(app) -> list[str]:
    return [n for (n,) in app.conn.execute("SELECT name FROM checkpoints")]


def test_a_fused_judge_and_a_judge_state_walk_one_machine(walk):
    """The composition, end to end: WORK fuses, RULE is a state, and the walk finishes."""
    snap, _ = walk
    assert snap.result is not None, "the machine did not reach a result"
    assert snap.result["path"] == ["work", "review", "rule", "work", "review", "rule"], snap.result
    assert snap.result["verdicts"] == [
        "done",
        "gathered",
        "breach",
        "done",
        "gathered",
        "approved",
    ], snap.result


def test_a_fused_judge_authors_a_distinct_id_at_every_visit(app, walk):
    """Two visits, two rows, and a guard that keeps the assertion from being vacuous.

    WORK is visited twice. A judge without coordinates would compose one `event_id` for both, and
    the second row would be refused and lost. Both are on the record, and they differ by the
    visit the judge learns from its own `Ctx`."""
    notes = [e for e in ledger_ids(app) if e.startswith("note:")]
    assert notes, "the fused judge appended nothing — the rest of this test is vacuous"
    assert notes == ["note:fs-1,work,0", "note:fs-1,work,3"], notes
    assert len(set(notes)) == len(notes)


def test_a_judge_state_is_addressed_by_its_own_state(app, walk):
    """A judgement with an address: its ops carry its own `state:` frame, not its predecessor's.

    Fused or split, a judge's ops were always framed — `scoped` is applied by the handler, so no
    author threads a scope through a callback. What a STATE buys is separation by NAME: two
    judgements under one state are told apart by the engine's positional `#N`, where these are
    distinct by construction."""
    ruled = [k for k in keys(app) if "state:rule;" in k]
    assert ruled, "the judge-state minted no ops — the rest of this test is vacuous"
    assert all(k.startswith("d:") for k in ruled), ruled
    rulings = [e for e in ledger_ids(app) if e.startswith("ruling:")]
    assert rulings == ["ruling:fs-1,rule,2", "ruling:fs-1,rule,5"], rulings


def test_incoming_carries_the_previous_report_so_nothing_is_measured_twice(walk):
    """`Ctx.incoming` is what makes the split affordable.

    Without it a judge-state's only route to its predecessor's measurement is to re-run the
    measurement — a second, differently-keyed op, and for a model judge a second non-deterministic
    call whose answer may differ from the one being judged. RULE reads the gate's result through
    `incoming` and never calls it, so the gate is called once per WORK and once per REVIEW plus
    once in the postamble."""
    _, deployment = walk
    assert deployment.gate_calls == 5, deployment.gate_calls


def test_the_carried_record_is_the_predecessors_own_measurement(walk):
    """Not merely that `incoming` carries a record, but that it carries the RIGHT one.

    `summary` is prose every `Report` has; `measured` is the half the record parameter types, and
    it is why `Ctx` is generic over the record at all. The gate stamps each call, so what RULE
    reads names the measurement it came from: visit 2 must see REVIEW's from visit 1 (the 2nd gate
    call) and visit 5 must see visit 4's.

    Measured rather than asserted: make the trampoline keep its FIRST report instead of the
    latest, and the walk still completes with the right path, the right verdicts and the same
    content-addressed artifact — every outcome assertion in this file and in the conformance lane
    stays green, and only this one and its durable twin redden. An outcome cannot tell a machine
    that carried its predecessor's measurement from one that carried a stale copy."""
    _, deployment = walk
    assert deployment.carried_outputs == [
        "clean at visit 1",
        "clean at visit 4",
    ], deployment.carried_outputs


def test_a_singleton_fibre_still_routes(walk):
    """REVIEW's fibre has one member, which is what a worker-state's fibre becomes once its
    judgement moves out. It is still a fibre: `route` has an arm for it, and dropping that arm is
    a `ty` error rather than a runtime surprise."""
    snap, _ = walk
    assert snap.result is not None
    turns = list(zip(snap.result["path"], snap.result["verdicts"], strict=True))
    gathered = [state for state, verdict in turns if verdict == "gathered"]
    assert gathered == ["review", "review"], turns


# --- Ctx stays hashable by its coordinates -----------------------------------------
#
# A `Report` embedded in `Ctx` would make a `Ctx` CONDITIONALLY unhashable: fine until a state
# produces a tree, then `TypeError`, so it fails only on the walk that happens to produce one.
# Pinned as the CLASS: a `Ctx` hashes by its coordinates, whatever it carries.


def _ctx_carrying(incoming: Report | None) -> Ctx[RulingState]:
    return Ctx(
        run_id=Segment("fs-1"), goal="g", state=RulingState.WORK, visit=1, incoming=incoming
    )


@pytest.mark.parametrize(
    ("case", "incoming"),
    [
        ("nothing yet — visit 0", None),
        ("a verdict alone", Report(WorkVerdict.DONE)),
        ("a tree, which is the shape that broke it", Report(WorkVerdict.DONE, tree={"a.py": "x"})),
        ("an empty tree, which is NOT the same as None", Report(WorkVerdict.DONE, tree={})),
        (
            "a measured record too",
            Report(
                WorkVerdict.DONE,
                summary="s",
                detail="d",
                measured=CommandRun(exit_code=0, failures=(), output=""),
                tree={"a.py": "x"},
            ),
        ),
    ],
)
def test_a_ctx_hashes_whatever_it_carries(case: str, incoming: Report | None):
    assert isinstance(hash(_ctx_carrying(incoming)), int), case


def test_the_coordinates_are_the_identity_and_eq_is_not_weakened():
    """The two halves of the fix, which a `hash=False` could get half right.

    Excluding `incoming` from the hash must not quietly exclude it from equality: two contexts
    differing only in what they carry are NOT equal, and unequal objects sharing a hash is exactly
    what a hash is permitted to do."""
    bare = _ctx_carrying(None)
    carrying = _ctx_carrying(Report(WorkVerdict.DONE, tree={"a.py": "x"}))
    assert hash(bare) == hash(carrying), "the coordinates are the identity"
    assert bare != carrying, "equality still sees what the context carries"
    assert hash(bare) != hash(
        _ctx_carrying(None).__class__(
            run_id=Segment("fs-1"), goal="g", state=RulingState.REVIEW, visit=1, incoming=None
        )
    ), "a different coordinate is a different context"


# --- the lift is structural, so the copy cannot fall behind the type -----------------------


def test_a_report_has_a_home_for_every_evidence_field():
    """`Report.of` copies what `Evidence` declares, so `Report` must be able to receive it.

    Pinned as a set relation rather than a field list: adding a field to `Evidence` and forgetting
    `Report` is exactly the drift the hand-written copy allowed."""
    evidence_fields = {f.name for f in fields(Evidence)}
    report_fields = {f.name for f in fields(Report)}
    assert evidence_fields <= report_fields, evidence_fields - report_fields


def test_the_lift_copies_every_evidence_field():
    """A PRODUCT over `Evidence`'s fields, not a list of the four that exist today.

    Deleting three of `fuse`'s four hand-written assignments left every other test green.
    This iterates the type, so a new field is covered on the day it is declared and a
    dropped copy reddens here."""
    sentinels = {
        "summary": "SUMMARY-SENTINEL",
        "detail": "DETAIL-SENTINEL",
        "measured": CommandRun(exit_code=7, failures=("f",), output="OUT-SENTINEL"),
        "tree": {"sentinel.py": "TREE-SENTINEL"},
    }
    assert sentinels.keys() == {f.name for f in fields(Evidence)}, (
        "a field was added to Evidence — give it a distinct sentinel here"
    )
    evidence = Evidence(**sentinels)
    report = Report.of(WorkVerdict.DONE, evidence)
    assert report.verdict is WorkVerdict.DONE
    for name in sentinels:
        assert getattr(report, name) == getattr(evidence, name), name


@pytest.mark.parametrize(
    ("kind", "make"),
    [
        (
            "Ctx",
            lambda: Ctx(
                run_id=Segment("fs-1"),
                goal="g",
                state=RulingState.WORK,
                visit=1,
                incoming=Report(WorkVerdict.DONE, tree={"a.py": "x"}),
            ),
        ),
        ("Report", lambda: Report(WorkVerdict.DONE, tree={"a.py": "x"})),
        ("Evidence", lambda: Evidence(summary="s", tree={"a.py": "x"})),
    ],
)
def test_the_whole_carrying_family_stays_hashable(kind: str, make: Callable[[], object]):
    """The `Ctx` fix generalised.

    All three are frozen dataclasses exported from `effective.machine`, all three carry a
    `Mapping`, and a property proved for ONE member of a set the codebase enumerates is not
    proved. The assertion is that `hash` does not RAISE — `hash` always returns an `int`, so
    `isinstance(..., int)` would name the wrong property."""
    hash(make())
