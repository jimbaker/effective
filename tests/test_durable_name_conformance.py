"""Every durable NAME in the store is in the language: the gate that reads the store.

The key lints scan SOURCE. A name composed correctly and then CORRUPTED on its way out, by a
foreign runtime doing its own string work below our boundary, is invisible to them and to a green
suite: replay re-derives the same corruption, so every durable test passes while the bytes on disk
are not in the grammar at all. This file asks the store what is actually there.

The shape it catches is a `Key` repr in a checkpoint name
(`Key(_value='step:assess_request', _scope=None)#2`): opacity fails loudly where a name is USED,
except in an append-only write, which is silent and permanent. The SDK's own `$awaitEvent:`
bookkeeping is in the language, read by `grammar._parse_foreign`.

A SQLite ctx needs no adapter, so the Absurd leg is the one that can fail on an adapter defect,
and the SQLite leg is the cheap everywhere-runnable half.

Role: conformance, one property on both engines.
"""

import sqlite3
from pathlib import Path

import pytest

from effective import Effect, append_ledger, call_tool, scoped, step, store_artifact
from effective.domain import CallTool
from effective.keys import Key, Segment, compose_key
from effective.keys.grammar import KeySyntaxError, parse, split_occurrence
from effective.ops import LedgerRow

pytestmark = pytest.mark.conformance

# No exclusions. The SDK's own suspension bookkeeping (`$awaitEvent:{our whole key}`,
# `$awaitTaskResult:{uuid}`) parses through `grammar._parse_foreign` and round-trips
# byte-exactly, so the gate covers the whole column: there is no name the engine writes that we
# cannot read.


def _offenders(names) -> list[str]:
    """Every name that is not in the language. No exclusions: that is the property."""
    out = []
    for name in names:
        if name is None:
            continue
        try:
            parse(name)
        except KeySyntaxError:
            out.append(name)
    return out


def _repeats_a_step_name() -> Effect[str]:
    """Two asks of ONE name, so the engine's duplicate-occurrence rule fires.

    That is the shape that corrupts: occurrence 1 is passed through and binds fine (psycopg's
    dumper renders `stored()`), while occurrence 2 goes through `f"{name}#{count}"` INSIDE the SDK
    — in Python, before any parameter is bound — so a `Key` there becomes a repr. A workflow that
    never repeats a name cannot detect this, which is why the fixture repeats one.
    """
    yield from call_tool("dup", {}, str)
    yield from call_tool("dup", {}, str)
    return (yield from step("done", CallTool(name="done", result_schema=str)))


def test_sqlite_writes_only_names_that_are_in_the_language(tmp_path: Path, sqlite_app) -> None:
    """The 0↔1 engine. Cheap, infra-free, and it runs everywhere."""
    from effective.handlers.absurd import DurableHandler

    db = tmp_path / "task.db"
    app = sqlite_app(str(db))

    @app.register_task("names")
    def _names(params, ctx):
        return DurableHandler(ctx, _CannedDomain()).run(_repeats_a_step_name)

    task_id = app.spawn("names", {})
    app.run_until_result(task_id)

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM checkpoints")]
        waits = [r[0] for r in conn.execute("SELECT waiting_event FROM tasks")]
    finally:
        conn.close()

    assert names, "the workflow committed no checkpoints, so this test proved nothing"
    assert _offenders(names) == [], f"checkpoint names not in the language: {_offenders(names)}"
    assert _offenders(waits) == []
    # The occurrence the SDK-side corruption rides on is present, so the fixture is doing its job.
    assert any(n.endswith("#2") for n in names), names


def test_no_durable_name_anywhere_carries_a_Key_REPR(tmp_path: Path, sqlite_app) -> None:
    """The specific corruption, asserted as its own property rather than folded into the above.

    Separate because it names the CULPRIT where the in-language check names only a symptom. Both
    fire on a repr today — measured: `$awaitEvent:Key(_value='child-done:x', _scope=None)` is
    refused by `parse`, because the repr is not a well-formed atom. But that is a coincidence of
    what a repr happens to contain, not a property: widen an atom's charset, or let a repr appear
    somewhere the parser is lenient, and the in-language check goes quiet while the corruption
    stays. A `Key` has no `__str__`, so this substring can only come from an f-string over one,
    and that is worth asserting on its own terms.
    """
    from effective.handlers.absurd import DurableHandler

    db = tmp_path / "task.db"
    app = sqlite_app(str(db))

    @app.register_task("names")
    def _names(params, ctx):
        return DurableHandler(ctx, _CannedDomain()).run(_repeats_a_step_name)

    app.run_until_result(app.spawn("names", {}))

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        every = [r[0] for r in conn.execute("SELECT name FROM checkpoints")]
        every += [r[0] for r in conn.execute("SELECT waiting_event FROM tasks") if r[0]]
        every += [r[0] for r in conn.execute("SELECT name FROM events")]
    finally:
        conn.close()

    assert every
    assert [n for n in every if "Key(_value" in n] == []


def test_a_stack_of_wrappers_is_STILL_adapted() -> None:
    """A wrapper stack over a raw SDK ctx is adapted, pinned at the seam rather than durably.

    `_adapt_ctx` walks to the innermost raw ctx at any depth; an `isinstance` check on the outer
    object alone would leave a wrapper stack unadapted and hand the SDK `Key`s. Asserted here with
    a fake SDK ctx, so the property is pinned even where Postgres is absent.
    """
    from effective.handlers import absurd as A

    sdk = pytest.importorskip("absurd_sdk")

    class _Raw(sdk.TaskContext):  # a raw SDK ctx, exactly what the SDK hands a task
        def __init__(self) -> None:
            self.seen: list[object] = []

    class _Wrapper:
        def __init__(self, inner) -> None:
            self._ctx = inner

    raw = _Raw()
    stack = _Wrapper(_Wrapper(raw))

    adapted = A._adapt_ctx(stack)

    assert adapted is stack, "the caller's own stack is returned, adapted in place"
    assert isinstance(stack._ctx._ctx, A.SdkCtx), "the innermost raw ctx must be wrapped"
    assert stack._ctx._ctx._ctx is raw

    # IDEMPOTENT: re-adapting must not double-wrap, or `.stored()` would run twice.
    A._adapt_ctx(stack)
    assert isinstance(stack._ctx._ctx, A.SdkCtx)
    assert stack._ctx._ctx._ctx is raw

    # ANTI-VACUITY: a bare raw ctx is still adapted directly, and a stack with no SDK ctx in it
    # is returned untouched — so this is not passing because the walk adapts everything.
    assert isinstance(A._adapt_ctx(raw), A.SdkCtx)
    plain = _Wrapper(_Wrapper(object()))
    assert A._adapt_ctx(plain) is plain
    assert not isinstance(plain._ctx._ctx, A.SdkCtx)


class _CannedDomain:
    """Answers every op without touching a model or a tool."""

    def run(self, op):
        return "ok"

    def run_metered(self, op):
        from effective.cost import Usage

        return "ok", Usage()


# --- the load-bearing leg: the REAL engine ---------------------------------------------------


def _pg() -> bool:
    from _conformance import pg_ready

    return pg_ready()


@pytest.mark.skipif(not _pg(), reason="no local Postgres with Absurd + ledger")
def test_absurd_writes_only_names_that_are_in_the_language_THROUGH_A_WRAPPER_STACK() -> None:
    """The leg that catches an unadapted wrapper stack, which SQLite structurally cannot.

    A SQLite ctx needs no adapter, so the SQLite legs above pass whether `_adapt_ctx` walks the
    stack or not. This one drives the vendored SDK through a WRAPPER STACK, the shape many test
    files build and the shape that writes `Key(_value='step:assess_request', _scope=None)#2` into
    `absurd.c_default` when the walk is missing.

    Scoped to THIS task's rows, not the whole table: a shared store carries other runs' rows, and
    a whole-table assertion would be a migration demand rather than a regression gate.
    """
    import psycopg
    from _conformance import PG_DSN

    from effective.absurd_worker import absurd_worker
    from effective.handlers.absurd import DurableHandler

    class _Wrapper:
        """One of ours, over the raw SDK ctx: the stack `_adapt_ctx` must walk.

        It DECLARES the three protocol methods rather than leaning on `__getattr__`, and that is
        not boilerplate: it is why `ty` cannot catch this class of defect. A wrapper that declares
        them satisfies `TaskContext` structurally, so the type checker is satisfied while a raw SDK
        ctx sits underneath receiving `Key`s. `FaultCtx`, the shared harness behind many of these
        stacks, is shaped exactly this way. A double that cheated with `__getattr__` would be
        REJECTED by `ty` and would therefore not reproduce the real hazard.
        """

        def __init__(self, inner) -> None:
            self._ctx = inner

        def step(self, name, thunk, /):
            return self._ctx.step(name, thunk)

        def await_event(self, name, /):
            return self._ctx.await_event(name)

        def sleep_until(self, when, /, *, name=None):
            return self._ctx.sleep_until(when, name=name)

        def __getattr__(self, attr):
            return getattr(self._ctx, attr)

    app = absurd_worker(PG_DSN)

    @app.register_task("name-conformance")
    def _names(params, ctx):
        return DurableHandler(_Wrapper(_Wrapper(ctx)), _CannedDomain()).run(_repeats_a_step_name)

    task_id = app.spawn("name-conformance", {})["task_id"]
    for _ in range(8):
        app.work_batch()

    with psycopg.connect(PG_DSN, connect_timeout=5) as conn:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT checkpoint_name FROM absurd.c_default WHERE task_id = %s", (task_id,)
            )
        ]

    assert names, "the task committed no checkpoints, so this proved nothing"
    assert [n for n in names if "Key(_value" in n] == [], names
    assert _offenders(names) == [], f"not in the language: {_offenders(names)}"
    # The duplicate-occurrence rule fired, which is the path that corrupts.
    assert any(n.endswith("#2") for n in names), names


# --- step 4: round-trip as a PROPERTY, and fidelity THROUGH the store -------------------------


class _CapturingCtx:
    """Records the `Key` the substrate hands down, at the boundary, before the engine sees it.

    The oracle question for fidelity is "did what we composed reach the store", and answering it
    needs the composed value — not a re-derivation, which would agree with the code by
    construction. This sits ABOVE the engine adapter (`_adapt_ctx` inserts `SdkCtx` *beneath* it,
    which the wrapper-stack pin above covers), so it sees `Key` objects rather than text.

    **The RecordingHandler is not the oracle.** It returns `step;tool:dup` twice for a repeated
    name where both durable engines write `step;tool:dup` and `step;tool:dup#2`: it consumes its
    trace positionally and never names an occurrence. Using it as the expected value would
    quietly assert that the engines must NOT disambiguate, the opposite of the rule.
    """

    def __init__(self, inner) -> None:
        self._ctx = inner
        self.handed: list[Key] = []

    def step(self, name, thunk, /):
        self.handed.append(name)
        return self._ctx.step(name, thunk)

    def await_event(self, name, /):
        self.handed.append(name)
        return self._ctx.await_event(name)

    def sleep_until(self, when, /, *, name=None):
        if name is not None:
            self.handed.append(name)
        return self._ctx.sleep_until(when, name=name)

    def __getattr__(self, attr):
        return getattr(self._ctx, attr)


def _shape_space() -> Effect[str]:
    """One workflow spanning the shapes a durable name can take.

    Chosen so a corruption anywhere in the family shows up: a bare author name (a COORDINATE of
    the arm), a structured one (spliced TERMS), the same name twice (the engine's occurrence rule
    — the path that produced 28 reprs), an artifact (a `/` path plus an algorithm-prefixed
    digest), a ledger id, and a `scoped(...)` frame.
    """
    yield from call_tool("dup", {}, str)
    yield from call_tool("dup", {}, str)
    yield from call_tool("s3-sync", {}, str)
    yield from store_artifact("blob", "text/plain")
    yield from append_ledger(LedgerRow(event_id=compose_key(t"reviewed:{Segment('m1')}")))
    yield from scoped(
        compose_key(t"rec:{0}"),
        lambda: step("inner", CallTool(name="inner", result_schema=str)),
    )
    return "done"


def _assert_fidelity(handed: list[Key], stored: list[str]) -> None:
    """Every stored name is the handed name, modulo the occurrence the ENGINE assigns.

    The occurrence is stripped with `grammar.split_occurrence` rather than a regex, so this
    exercises the one reader instead of adding a second — the defect the bridges still carry.
    """
    assert stored, "nothing was written, so this proved nothing"
    assert [split_occurrence(s)[0] for s in stored] == [k.stored() for k in handed]
    # ANTI-VACUITY: the repeat really did get disambiguated, so the strip above did work.
    assert any(s.endswith("#2") for s in stored), stored
    # and the suffix is the ONLY difference — no name was silently rewritten
    assert stored != [k.stored() for k in handed]


def test_sqlite_stores_EXACTLY_the_names_the_substrate_handed_down(
    tmp_path: Path, sqlite_app
) -> None:
    """Fidelity, 0↔1. A well-formed name that is the WRONG name passes the in-language gate."""
    from effective.handlers.absurd import DurableHandler

    db = tmp_path / "task.db"
    app = sqlite_app(str(db))
    captured: dict[str, _CapturingCtx] = {}

    @app.register_task("shapes")
    def _shapes(params, ctx):
        captured["ctx"] = _CapturingCtx(ctx)
        return DurableHandler(captured["ctx"], _CannedDomain()).run(_shape_space)

    app.run_until_result(app.spawn("shapes", {}))

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        stored = [r[0] for r in conn.execute("SELECT name FROM checkpoints ORDER BY rowid")]
    finally:
        conn.close()
    _assert_fidelity(captured["ctx"].handed, stored)


@pytest.mark.skipif(not _pg(), reason="no local Postgres with Absurd + ledger")
def test_absurd_stores_EXACTLY_the_names_the_substrate_handed_down() -> None:
    """The same property on the engine that actually corrupted, through a wrapper stack."""
    import psycopg
    from _conformance import PG_DSN

    from effective.absurd_worker import absurd_worker
    from effective.handlers.absurd import DurableHandler

    app = absurd_worker(PG_DSN)
    captured: dict[str, _CapturingCtx] = {}

    @app.register_task("shape-fidelity")
    def _shapes(params, ctx):
        captured["ctx"] = _CapturingCtx(ctx)
        return DurableHandler(captured["ctx"], _CannedDomain()).run(_shape_space)

    task_id = app.spawn("shape-fidelity", {})["task_id"]
    for _ in range(8):
        app.work_batch()

    with psycopg.connect(PG_DSN, connect_timeout=5) as conn:
        stored = [
            r[0]
            for r in conn.execute(
                "SELECT checkpoint_name FROM absurd.c_default WHERE task_id = %s "
                "ORDER BY updated_at, checkpoint_name",
                (task_id,),
            )
        ]
    _assert_fidelity(captured["ctx"].handed, stored)


def test_every_foreign_name_the_SDK_can_MINT_is_one_we_can_READ() -> None:
    """Scan the vendored SDK for the names it composes, rather than trusting a list of two.

    The rest of this file asks the store what is there, which is the strongest question available
    but only reaches what has already been written. `$awaitTaskResult:` is written by
    `TaskContext.await_task_result`, which this repo does not call — so it has **zero** rows, and
    every store-reading assertion here passes without ever meeting it. A gate is bounded by what
    it SCANS, and the store does not scan an unexercised code path.

    So this reads the producer instead. `absurd.sql` is vendored under `infra/absurd/`; the
    PYTHON SDK this reads is a PyPI package in `.venv`, co-versioned with it at 0.5.0 through
    `PIN.txt` plus `uv.lock` and an `exclude-newer-package` override. Pinned either way, which is
    what makes its set of foreign mints closed and readable rather than an open denylist — a
    version bump that adds a third fails here instead of silently widening the language.
    """
    import ast
    import re

    import absurd_sdk

    source = Path(absurd_sdk.__file__).read_text()

    # Two sweeps, because the narrow one is only trustworthy if the wide one agrees. The wide
    # sweep walks the AST and reads every string CONSTANT containing `$` — including each chunk
    # of an f-string — so a mint spelled some other way cannot hide from it by not matching a
    # source regex. A regex over the text was tried first and four forms escaped it silently:
    # a `SIGIL = "$"` interpolated in, implicit concatenation, a `\x24` escape, and `"$" + "..."`
    # built then `.format`ted. Under the AST sweep the first and last surface as a bare `"$"`,
    # which breaks the equality below rather than passing quietly.
    every_marker = {
        found
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "$" in node.value
        for found in re.findall(r"\$\w*", node.value)
    }
    assert every_marker == {"$awaitEvent", "$awaitTaskResult"}, (
        f"the vendored SDK names a foreign marker this file has never seen: {every_marker}"
    )

    minted = set(re.findall(r'f"(\$\w+):\{(\w+)\}"', source))
    assert minted == {("$awaitEvent", "event_name"), ("$awaitTaskResult", "task_id")}, (
        f"the vendored SDK mints a foreign name this file has never seen: {minted}. "
        "Read what it interpolates, then decide whether it is one of our keys (like "
        "`$awaitEvent`) or the other runtime's own atom (like `$awaitTaskResult`)."
    )

    # And both readings are exercised here, on names shaped exactly as the SDK writes them.
    wraps_our_key = parse("$awaitEvent:review:m1")
    assert wraps_our_key.wrapped is not None
    assert wraps_our_key.wrapped.render() == "review:m1"

    for nibble in "0123456789abcdef":
        carries_their_atom = parse(f"$awaitTaskResult:{nibble}198f0c1-1234-7abc-8def-0123456789ab")
        assert carries_their_atom.wrapped is None, "a task id is Absurd's atom, not a term of ours"
