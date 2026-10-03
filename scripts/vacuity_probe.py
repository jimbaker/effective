"""What a KEY actually did during a test — observed at the store, not inferred from source.

**The question.** A key is written down in a test, composed or spelled, and rides the durable
trampoline. Did it *do* anything? Much test usage of `Key` is deliberately inert — construct one,
take it apart, assert on the algebra — and that is fine, it is a unit test of the key type. What
this probe looks for is different: a key that goes through the store and yet nothing keyed off
it. It acted in a **logging capacity**, and replay is not about logging.

**Three seams carry a key, and only one of them can let it select a value:**

| seam | what it sees | wrong key changes a VALUE? |
|---|---|---|
| **store** — `SqliteTaskContext` | a checkpoint write/hit/miss | **yes** — the key IS the index |
| **replay** — `ReplayHandler` | the key, compared to a recorded one | no — it can only abort |
| **domain** — a `DomainLayer` | the tool's name and arguments | no — it never sees a key |

`sqlite.py:414` is `SELECT state FROM checkpoints WHERE task_id={task_id} AND name={name}`: two
keys colliding means the second step reads the first's state. `replay.py:98` is
`expected = self.recorded[self._i]` followed by an equality check — the value comes from a
*position*, so `ReplayHandler` is structurally incapable of letting a key select anything.

**So this probe attaches at the store seam, and it observes rather than mutates.** Per test it
records three events:

- ``write``     — the thunk ran and a checkpoint row was inserted.
- ``hit``       — a checkpoint row was found and the thunk was **skipped**.
- ``event_hit`` — an event lookup found a payload.
- ``miss``      — an event lookup found nothing.
- ``await``     — the key was used as a park address (`await_event`/`repark`).
- ``emit`` / ``idempotency`` — selective seams this perturbation cannot judge (see below).

``hit`` and ``event_hit`` are the two that prove an identity did work: in both, the key SELECTED
a stored value. Everything else is the key being written down.

and classifies each test by what its keys were allowed to do:

- ``resolving``  — at least one ``hit``. A key selected a value; distinctness is load-bearing.
- ``write-only`` — keys reached the store and none was ever consulted. **The vacuity candidate.**
- ``no-store``   — the test never reached the store at all.

**WHERE IT ATTACHES, AND WHY THAT LINE.** `SqliteTaskContext.step` qualifies a repeated name to
``name#N`` at `sqlite.py:411`, *before* the SELECT. A probe attached above that line sees one key
where the store holds two rows, so the occurrence count is read back off the ctx and recorded
beside the key. This is the same nominal-vs-structural trap the rest of the arc is about, one
level down.

**WHAT IT IS BLIND TO — a gate is bounded by what it SCANS.**

- **The SQLite engine only.** The Absurd/Postgres ctx is a different class and is not patched, so
  an Absurd-only test reads as ``no-store`` rather than being measured. Say "unscanned", never
  "clean".
- **Keys consumed above the store** — inside a layer, a projection, a dashboard read, or an
  assertion on a composed value — never cross this boundary and are invisible here.
- **`sleep_until` is nameless on SQLite** (`sqlite.py:456`): the wake time lands on
  `tasks.available_at` and no checkpoint row is written, so a sleep key cannot appear.
- **A `write-only` verdict is a CANDIDATE, not a defect.** A single-pass run that never re-runs a
  task has no opportunity for a hit, and that can be exactly the right test. The verdict says the
  test did not exercise the key's resolving role; a reader supplies the judgement.
- **THE PERTURBATION IS PROPERTY-PRESERVING, so a test asserting a PROPERTY cannot see it.**
  `SHIFT` produces a key a real scope could have minted — still in the language, still free of a
  `Key` repr — which is exactly why it is safe to apply, and exactly why
  `test_no_durable_name_anywhere_carries_a_Key_REPR` passes under it. That test is not inert; it
  asserts something the shift deliberately does not disturb. Read `key-inert` as *"neither
  consulted nor spelling-observed"*, which is what was measured, and never as *"tests nothing"*.

**THE VERDICT NAMES WHAT WAS MEASURED, AND THE READER JUDGES** — the same division
`scripts/key_sweep.py` makes. This bucket was called `VACUOUS` for one afternoon and the name was
wrong: it asserted a conclusion the instrument had not earned, and three of the first twenty-three
hits were tests that legitimately do not turn on a key at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "vacuity.jsonl"

# ---------------------------------------------------------------- the plugin


#: The injective perturbation. `Key.prefixed` is the substrate's OWN re-scoping operation
#: (`keys.py:387`) and `scope_prefix` produces exactly this `{atom};` shape, so a constant
#: prefix maps the key space onto a shifted copy of itself: a bijection, delimiters intact,
#: and the result is a key a real scope could have minted. Distinctness is PRESERVED, so
#: every checkpoint still resolves — only a reader comparing a *stored spelling* against a
#: composed one can notice. That is the whole discrimination.
#:
#: **IT WAS `probe:0/` FIRST, AND THAT WAS OUT OF THE LANGUAGE.** `scope_prefix`'s own summary
#: line said the prefix is `{atom}/` while `TERM_SEPARATOR` is `;` and its code emits `;` —
#: a docstring contradicting its function, trusted instead of run. The whole first full-suite
#: pass used an unparseable prefix, which reddens every grammar assertion for the
#: INSTRUMENT's reason and inflates `spelling-observed`. Caught by this module's own pin,
#: not by review; the docstring is fixed in the same commit.
SHIFT = "probe:0;"


def _shifter(active: bool):
    """The perturbation, TYPE-PRESERVING and IDEMPOTENT.

    `Key.prefixed(SHIFT)` and `SHIFT + key.stored()` render the same text, so a
    `str`-typed seam stays consistent with a `Key`-typed one; a seam handed a `Key` keeps
    getting a `Key`, because changing the type and the value at once
    would make a failure ambiguous between them.

    Idempotence is not tidiness. `await_event` delegates to `self.peek_event`, which is the
    patched wrapper, so a naive shift applies twice and stores `probe:0;probe:0;…`. Seams that
    call each other are the normal case, so the shift has to be a projection."""

    def shift(name):
        if not active or _text(name).startswith(SHIFT):
            return name
        prefixed = getattr(name, "prefixed", None)
        return prefixed(SHIFT) if callable(prefixed) else SHIFT + name

    return shift


def _text(name: object) -> str:
    """The stored spelling of a name that may be a `Key` or already text.

    **Both arrive, and finding that out was the point.** `SqliteTaskContext.emit_event`
    annotates `name: str` and is handed a `Key` by at least one caller — an unenforced
    annotation is a claim about a value, never the value. Unwrapping through `.stored()`
    rather than `str()` matters for the same reason it does everywhere else here: `str(key)`
    is the repr, and a repr in a durable name is the defect this arc keeps finding."""
    stored = getattr(name, "stored", None)
    return stored() if callable(stored) else str(name)


class StoreProbe:
    """A pytest plugin that watches the SQLite store seam and writes one row per event.

    With ``perturb_keys``, it also applies `SHIFT` to every name the store sees. The store
    stays internally consistent, so a test only fails if it OBSERVES a spelling."""

    def __init__(self, out: Path, *, perturb_keys: bool = False) -> None:
        self.out = out
        self.current: str | None = None
        self.rows: list[dict] = []
        self.perturb_keys = perturb_keys
        self.outcomes: dict[str, str] = {}

    # -- pytest hooks ---------------------------------------------------

    def pytest_configure(self, config) -> None:
        self._install()

    def pytest_runtest_logstart(self, nodeid: str, location) -> None:
        self.current = nodeid

    def pytest_runtest_logfinish(self, nodeid: str, location) -> None:
        self.current = None

    def pytest_runtest_logreport(self, report) -> None:
        """Record the OUTCOME too — under perturbation, pass-vs-fail is the finding."""
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            self.outcomes[report.nodeid] = report.outcome

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("w") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")
        outcomes = self.out.with_suffix(".outcomes.json")
        outcomes.write_text(json.dumps(self.outcomes, indent=1))

    # -- the instrument -------------------------------------------------

    def _record(self, key: object, occurrence: int, event: str) -> None:
        if self.current is None:  # a fixture or collection-time call, outside any test
            return
        self.rows.append(
            {"test": self.current, "key": _text(key), "occurrence": occurrence, "event": event}
        )

    def _install(self) -> None:
        """Patch every name-bearing seam the SQLite store has — SIX, and the set was ENUMERATED
        from the classes rather than guessed, after guessing wrong twice.

        The misses were found by the probe's own output, never by reading it.
        ``…never_alias_two_idempotency_keys_onto_one_task`` came back inert, which is not
        credible for a test about aliasing — `idempotency_key` lives on `SqliteApp.spawn`, not
        on the ctx. Then four `test_parked_reader.py` cases came back inert while visibly
        asserting on a park's `wake_event` — because `await_event` raises `_Suspend(event=name)`
        with the name it was HANDED, so the park record kept an unperturbed spelling while the
        lookup used a shifted one. A gate is bounded by what it SCANS, this one included, and
        the honest way to bound it is `inspect.signature` over the class rather than a list.

        `sleep_until` is deliberately absent: a SQLite sleep is NAMELESS (`sqlite.py:456` — the
        wake time lands on `tasks.available_at` and no row is keyed), so there is nothing to
        perturb. `register_task`/`spawn`'s `name` is a TASK name, a different namespace from a
        key, and is left alone."""
        from effective.sqlite import SqliteApp, SqliteTaskContext

        probe = self
        shift = _shifter(self.perturb_keys)
        original_step = SqliteTaskContext.step
        original_await = SqliteTaskContext.await_event
        original_peek = SqliteTaskContext.peek_event
        original_ctx_emit = SqliteTaskContext.emit_event
        original_repark = SqliteTaskContext.repark
        original_app_emit = SqliteApp.emit_event
        original_spawn = SqliteApp.spawn

        text_shift = shift

        def step(self, name, thunk, /):
            ran: list[bool] = []

            def watched():
                ran.append(True)
                return thunk()

            result = original_step(self, shift(name), watched)
            # read the occurrence back off the ctx: `step` bumps it before the SELECT, so
            # after the call this is the count that qualified the stored name.
            occurrence = self._occurrences.get(shift(name), 1)
            probe._record(name, occurrence, "write" if ran else "hit")
            return result

        def await_event(self, name, /):
            # Patched IN ADDITION to `peek_event`, because the park record is written from the
            # name this method raises with, not the one the lookup used.
            probe._record(name, 1, "await")
            return original_await(self, shift(name))

        def peek_event(self, name, /):
            found, payload = original_peek(self, shift(name))
            probe._record(name, 1, "event_hit" if found else "miss")
            return found, payload

        def repark(self, name, /):
            probe._record(name, 1, "await")
            return original_repark(self, shift(name))

        def ctx_emit_event(self, name, payload, /, **kw):
            probe._record(name, 1, "emit")
            return original_ctx_emit(self, text_shift(name), payload, **kw)

        def app_emit_event(self, name, payload):
            probe._record(name, 1, "emit")
            return original_app_emit(self, text_shift(name), payload)

        def spawn(self, name, params, max_attempts=3, *, idempotency_key=None):
            if idempotency_key is not None:
                probe._record(idempotency_key, 1, "idempotency")
                idempotency_key = text_shift(idempotency_key)
            return original_spawn(
                self, name, params, max_attempts, idempotency_key=idempotency_key
            )

        SqliteTaskContext.step = step
        SqliteTaskContext.await_event = await_event
        SqliteTaskContext.peek_event = peek_event
        SqliteTaskContext.repark = repark
        SqliteTaskContext.emit_event = ctx_emit_event
        SqliteApp.emit_event = app_emit_event
        SqliteApp.spawn = spawn


# ---------------------------------------------------------------- reporting

CLASSES = ("resolving", "unjudged-seam", "write-only", "no-store")


def classify(rows: list[dict]) -> dict[str, str]:
    """One verdict per test nodeid.

    ``unjudged-seam`` is the honest bucket, not a hedge. An ``emit`` address and an
    ``idempotency`` key are both selective — a wrong one loses a wake, a colliding one folds
    two tasks into one — but the INJECTIVE perturbation cannot tell whether that mattered,
    because a consistent shift preserves both properties. Judging those needs the
    distinctness-destroying mutation, which is specified and not built (see `main`'s epilog).
    Reporting them as vacuous would be the instrument claiming a domain it does not scan."""
    events: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        events[row["test"]][row["event"]] += 1
    verdicts = {}
    for test, counts in events.items():
        if counts["hit"] or counts["event_hit"]:
            verdicts[test] = "resolving"
        elif counts["emit"] or counts["idempotency"]:
            verdicts[test] = "unjudged-seam"
        elif counts["write"]:
            verdicts[test] = "write-only"
        else:
            verdicts[test] = "no-store"
    return verdicts


def report(rows: list[dict], *, by_file: bool) -> str:
    verdicts = classify(rows)
    lines: list[str] = []
    totals = Counter(verdicts.values())
    lines.append(
        f"{len(verdicts)} tests reached the probe — "
        + ", ".join(f"{totals[c]} {c}" for c in CLASSES)
    )
    lines.append("")
    if by_file:
        per_file: dict[str, Counter] = defaultdict(Counter)
        for test, verdict in verdicts.items():
            per_file[test.split("::")[0]][verdict] += 1
        width = max((len(f) for f in per_file), default=0)
        lines.append(f"{'file'.ljust(width)}  resolving  write-only  no-store")
        for path in sorted(per_file, key=lambda p: -per_file[p]["write-only"]):
            counts = per_file[path]
            lines.append(
                f"{path.ljust(width)}  {counts['resolving']:>9}  "
                f"{counts['write-only']:>10}  {counts['no-store']:>8}"
            )
    else:
        for test in sorted(verdicts, key=lambda t: (verdicts[t], t)):
            lines.append(f"  [{verdicts[test]}] {test}")
    return "\n".join(lines)


def _label(cls: str, *, observed: bool) -> str:
    """The four-way join of "did the store consult it" against "did the test observe it"."""
    match cls, observed:
        case "resolving", True:
            return "resolving+observed"
        case "resolving", False:
            return "resolving"
        case _, True:
            return "spelling-observed"
        case "unjudged-seam", False:
            return "unjudged-seam"
        case _:
            return "key-inert"


def verdict(rows: list[dict], perturbed: dict[str, str], baseline: dict[str, str]) -> str:
    """Join the observation pass to the perturbation pass — the two together are the finding.

    A key did work in a test if the store CONSULTED it (a ``hit``) or the test OBSERVED its
    spelling (perturbation reddens it). A test where neither holds ran its keys through the
    trampoline for nothing."""
    classes = classify(rows)
    lines: list[str] = []
    buckets: dict[str, list[str]] = defaultdict(list)
    for test, cls in classes.items():
        if baseline.get(test) != "passed":  # only judge tests that pass unperturbed
            continue
        buckets[_label(cls, observed=perturbed.get(test, "passed") != "passed")].append(test)
    order = (
        "key-inert",
        "unjudged-seam",
        "spelling-observed",
        "resolving",
        "resolving+observed",
    )
    lines.append(
        "  ".join(f"{len(buckets[b])} {b}" for b in order if buckets[b]) or "nothing measured"
    )
    lines.append("")
    per_file: dict[str, Counter] = defaultdict(Counter)
    for label, tests in buckets.items():
        for test in tests:
            per_file[test.split("::")[0]][label] += 1
    for path in sorted(per_file, key=lambda p: (-per_file[p]["key-inert"], p)):
        counts = per_file[path]
        if not counts["key-inert"]:
            continue
        lines.append(f"{path}  ({counts['key-inert']} key-inert of {sum(counts.values())})")
        for test in sorted(buckets["key-inert"]):
            if test.startswith(path + "::"):
                lines.append(f"    {test.split('::', 1)[1]}")
    return "\n".join(lines)


def _run(paths: list[str], *, perturb: bool, out: Path) -> int:
    import pytest

    code = pytest.main(
        [*paths, "-q", "--no-header", "--no-cov", "-p", "no:cacheprovider"],
        plugins=[StoreProbe(out, perturb_keys=perturb)],
    )
    return int(code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["tests/"], help="what to run")
    parser.add_argument("--report", action="store_true", help="report from a previous run")
    parser.add_argument("--by-file", action="store_true", help="aggregate per file")
    parser.add_argument(
        "--perturb", action="store_true", help=f"shift every stored key by {SHIFT!r}"
    )
    parser.add_argument(
        "--verdict", action="store_true", help="run BOTH passes and join them into a verdict"
    )
    args = parser.parse_args(argv)
    paths = args.paths or ["tests/"]
    shifted = OUT.with_name("vacuity-perturbed.jsonl")

    if args.verdict:
        if (code := _run(paths, perturb=False, out=OUT)) not in (0, 1):
            return code
        if (code := _run(paths, perturb=True, out=shifted)) not in (0, 1):
            return code
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        print(
            verdict(
                rows,
                json.loads(shifted.with_suffix(".outcomes.json").read_text()),
                json.loads(OUT.with_suffix(".outcomes.json").read_text()),
            )
        )
        return 0

    if not args.report:
        code = _run(paths, perturb=args.perturb, out=shifted if args.perturb else OUT)
        if code not in (0, 1):  # 1 == some tests failed, which is fine for observation
            print(f"pytest exited {code}", file=sys.stderr)
            return code

    source = shifted if args.perturb else OUT
    if not source.exists():
        print(f"no probe output at {source}", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    print(report(rows, by_file=args.by_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
