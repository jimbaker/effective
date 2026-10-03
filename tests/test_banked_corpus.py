"""The banked SQLite corpus as a COMPAT FIXTURE: the suite's only cross-version reader check.

Every other durable test writes a store and reads it back in the same process, with the same
code. That proves the reader agrees with *this* writer. It cannot answer the question a durable
substrate actually has to survive: **does today's reader still read what last month's writer
wrote?** Only stores written by an older revision can answer that.

The corpus spans a real key-grammar change. The eras, and what separates them, are described in
`agent.bank`, which both this module and the `just corpus-check` gate read so the two cannot
drift. The two eras are asserted differently and deliberately:

| era     | asserted                                                                         |
|---------|----------------------------------------------------------------------------------|
| current | readable by today's reader, and replayable with a model and sandbox that raise  |
| legacy  | unreadable, for the GRAMMAR's reason: `Key.parse` validates on the read boundary |

Replay re-derives keys by RE-EXECUTING the workflow, so it binds to whatever the writer's revision
composed; asserting that a legacy store replays would assert that the key grammar may never
change. Current-era stores must replay, and that is what makes them useful as the next era's
legacy fixture.

**WHY THE SKIPS ARE PER-ERA.** The corpus is gitignored, so it is whole where the campaign ran and
partial or absent anywhere else; one skip for the whole corpus would make assertions about the
checkout rather than the code. The asymmetry is named in `agent.bank`: a current-era corpus is
REPRODUCIBLE (`just rebank`, ~$0.005), so its absence is a repairable condition the gate fixes
before the suite runs; a legacy corpus is not reproducible by any command, because the writer
that wrote it is gone, so its absence is a fact about this machine that these tests state and
skip. A skip is silent, so its non-silent partner is `just corpus-check`, inside `just check`.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from agent.bank import (
    BANK_ARMS,
    BENCH_SEED,
    BENCHES,
    CURRENT_ERA,
    all_stores,
    era_root,
    is_current,
    read_stamp,
    status,
)
from effective.checkpoints import Checkpoint, keys, read_sqlite_task
from effective.keys.grammar import KeySyntaxError

pytestmark = pytest.mark.skipif(
    not BENCHES.exists(), reason="banked corpus absent (gitignored; `just rebank`)"
)

_STORES = all_stores()
_CURRENT = [p for p in _STORES if is_current(p)]
_LEGACY = [p for p in _STORES if not is_current(p)]

needs_current = pytest.mark.skipif(
    not _CURRENT,
    reason=f"no {CURRENT_ERA} stores — `just rebank`; `just corpus-check` does it in `just check`",
)
needs_legacy = pytest.mark.skipif(
    not _LEGACY,
    reason="no legacy stores — the cross-version half lives where the old campaign ran; "
    "no command can produce one, since the writer that wrote it is gone",
)


def _task_ids(path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute("SELECT DISTINCT task_id FROM checkpoints")]
    finally:
        conn.close()


def test_the_corpus_is_not_vacuous():
    """A fixture that silently matched zero stores would pass forever and check nothing — the
    vacuous-green failure mode the identity sweep already hit once (a pre-flight grep came back
    clean while two tests depended on the composed spelling, one of them failing vacuously).

    Note what this does NOT assert any more: that BOTH eras are present. That belongs to the marks
    above, which say which half of the fixture each machine can run. What stays an assertion is
    that a corpus tree holding no stores at all is broken rather than merely partial."""
    assert _STORES, "the corpus tree exists but holds no task.db — the fixture would be vacuous"


@needs_current
def test_the_current_era_carries_a_stamp():
    """The tie between the fixture and its gate: `just corpus-check` decides the corpus is usable
    by reading a stamp, so a current-era tree without one means the two disagree about what is
    here. It also catches a tree copied from another machine — the repair is the same command, and
    it is nearly free, since `contrast_run.py` resumes over clean cells and only writes the stamp.
    """
    stamp = read_stamp(era_root())
    assert stamp is not None, f"{era_root()} has no readable stamp — run `just rebank`"
    assert stamp.era == CURRENT_ERA
    assert stamp.seed == BENCH_SEED
    assert status().usable, f"the gate would not pass this corpus: {status().problems}"


@needs_current
def test_every_current_store_is_readable_by_todays_reader():
    """Today's reader opens today's stores.

    SCOPED TO THE CURRENT ERA: `Key.parse` validates, so a store written under an older grammar
    does not come back through the reader at all. The legacy half is pinned as unreadable BY
    DESIGN by the test below rather than left silent.

    A store is readable when the checkpoint reader opens it and returns typed `Checkpoint`s whose
    keys are non-empty. Deliberately NOT asserted: any particular key spelling. That is what
    changes, and pinning it here would turn this into a freeze on the grammar."""
    failures: list[str] = []
    read = 0
    for store in _CURRENT:
        for task_id in _task_ids(store):
            try:
                # `ty: ignore` — the documented exception to the `UUID` task-id seam: stores
                # banked before `SqliteApp.spawn` minted uuid7 carry the composed text
                # `f"{name}-{seq}"`, which `UUID(...)` cannot parse. A banked artifact is read as
                # it was written; the seam stays typed for every store the current engine writes.
                checkpoints = read_sqlite_task(store, task_id)  # ty: ignore[invalid-argument-type]
            except Exception as exc:
                failures.append(f"{store}: {type(exc).__name__}: {exc}")
                continue
            if not all(isinstance(c, Checkpoint) for c in checkpoints):
                failures.append(f"{store}: reader returned a non-Checkpoint")
            if any(not k for k in keys(checkpoints)):
                failures.append(f"{store}: an empty op key")
            read += 1
    assert not failures, f"{len(failures)} of {read} banked tasks unreadable:\n" + "\n".join(
        failures[:10]
    )


@needs_current
@needs_legacy
def test_a_legacy_store_is_unreadable_and_that_is_the_ruling():
    """A legacy store fails to read, and fails for the GRAMMAR's reason.

    `Key` holds terms, so `Key.parse` validates, so a key written under an older grammar raises
    rather than travelling as opaque text. A guarantee that does not hold is PINNED as not
    holding instead of being left silent.

    If the read boundary turns lazy, this goes red and names what changed. If the legacy corpus
    is re-banked, it goes red too, which is the correct signal that there is no second era."""
    reasons: list[str] = []
    for store in _LEGACY:
        for task_id in _task_ids(store):
            try:
                read_sqlite_task(store, task_id)  # ty: ignore[invalid-argument-type]
            except KeySyntaxError as exc:
                reasons.append(str(exc))
            except Exception:  # any other failure is not the claim being made
                pass
    assert reasons, (
        "every legacy store read cleanly — either the corpus lost its older eras, or the read "
        "boundary stopped validating, and the second would be a silent return to text-as-truth"
    )


@needs_current
def test_current_era_stores_still_replay():
    """The replayable half, as a TEST rather than a docstring.

    The bank-time `replay_ok` in each `trial.json` describes the code that banked the store, and
    says nothing about today's code; only a re-drive here does.

    `replay_combinator` re-drives the workflow over the completed task's checkpoints with a model
    that RAISES if called and a sandbox that RAISES if executed, so reaching the banked answer
    proves every op re-bound from the record rather than being recomputed. That is the poison-stub
    proof, and it is what makes these stores usable as the NEXT era's legacy fixture.

    Scoped to the `combinator` arm because its replay entry point needs only the task and its rows;
    the structured arm additionally needs the harness inputs. One arm exercises the same
    record/replay path, which is why `agent.bank.BANK_ARMS` banks that arm alone."""
    from agent.contrastbench import SIZES, make_log, make_tasks, replay_combinator

    tasks = {task.task_id: task for task in make_tasks(BENCH_SEED)}
    replayed = 0
    for trial_file in sorted(era_root().rglob("trial.json")):
        trial = json.loads(trial_file.read_text())
        if trial.get("arm") not in BANK_ARMS:
            continue
        task = tasks.get(trial["task_id"])
        if task is None:
            continue  # banked under a different generator seed; not this test's business
        rows, _ = make_log(BENCH_SEED, SIZES[trial["size"]])
        got = replay_combinator(task, rows, db_path=trial_file.parent / "task.db")
        assert got == trial["answer"], (
            f"{trial_file.parent}: replayed {got!r} != {trial['answer']!r}"
        )
        replayed += 1
    assert replayed, "no current-era combinator store replayed — the pin would be vacuous"
