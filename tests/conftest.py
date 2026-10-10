"""Which database a test run talks to, and what state it starts from.

Two regimes, and the choice is made by whether pytest-xdist is driving.

**Parallel — a database per worker.** The durable lane cannot share one database. Absurd is
pull-only: a test drains work by calling `work_batch()` a bounded number of times, and
`work_batch` claims *whatever is ready*, not that test's task. Two workers on one queue steal each
other's claims. A per-worker *queue* is not enough either: the `ledger` and the projections are
shared, and the reset below would truncate them under a sibling mid-test. So the isolation boundary
is the whole database: the controller copies a prepared
template once per worker, and each worker keeps saying `default` and `ledger` to its own copy.
The ~14 test modules that hardcode `absurd.*_default` need no change at all.

**Serial — one shared database, reset at session start.** A task left **non-terminal** by an
earlier run is not inert. It stays claimable, and every later test spends claims on it. Past a few
dozen, budgeted drains start missing their own task and the failure looks like a substrate bug:
`state='pending'` on a test that passes in isolation. Leftover `voi-*` runs parked on grants that
never arrive, and fork children dying on the way out, would starve `test_govern_durable` and
`test_skills_durable` cases that are green run alone.

The reset runs at session **start** rather than at exit on purpose — so the run you just debugged
is still on disk to inspect, while the run you are about to do starts from a queue nobody else is
holding. Non-terminal runs are cancelled, and terminal rows (`completed`/`failed`/`cancelled`)
are left alone: they cost nothing to skip and they are the audit trail. The parallel regime
inherits the same principle by dropping and recreating each worker database at session start.

**Between tests, in both regimes:** each test's non-terminal runs are cancelled at its teardown,
since a leftover spends a later test's `work_batch()` within one session too. A second serial run
on the database is refused, whatever it runs.

Cheap by construction, and completely silent when no Postgres is reachable: one guarded connect,
two seconds, then every backend-gated test skips through its own `pg_ready()`. That is the CI
path and the `test-core` path, and neither may be made to fail here.
"""

import os
from collections import Counter
from contextlib import ExitStack, closing
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

# Before `effective.sqlite` is imported by anything, so every connection this session opens is
# accounted for. `connect` reads it per call, but setting it late would still miss the
# import-time opens some modules do.
#
# Plain assignment, NOT `setdefault`: `connect` gates on `== "1"`, so an inherited
# `EFFECTIVE_TRACK_CONNECTIONS=0` would leave `unclosed_connections()` empty and the gate below
# would pass **vacuously, with no signal** — reproduced, a planted leak exited 0. A test session
# always tracks; the env var is for production, which does not run this file.
os.environ["EFFECTIVE_TRACK_CONNECTIONS"] = "1"

from effective.lint import configured
from effective.sqlite import SqliteApp, unclosed_connections

_DEFAULT_DSN = "postgresql://effective:effective@localhost:5432/effective"

# Read BEFORE the per-worker rewrite below: the template, the admin connection and every worker
# database are all named relative to the database the run was pointed at.
_BASE_DSN = os.environ.get("DATABASE_URL", _DEFAULT_DSN)


def _dbname(dsn: str) -> str:
    """The database name in a `postgresql://…/name` URL."""
    return urlsplit(dsn).path.lstrip("/")


def _named(dsn: str, dbname: str) -> str:
    """`dsn` pointed at a different database on the same server.

    URL form only, deliberately. psycopg's `make_conninfo` would be the general answer, but it
    emits keyword form (`host=… dbname=…`), and `effective.ledger.to_sqlalchemy_url` rewrites the
    `postgresql://` PREFIX by hand — so a keyword DSN would sail past every psycopg call and fail
    only where SQLModel gets involved. Every DSN in this repo is a URL, and `_require_url_dsn`
    below turns that from a wish into a check."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{dbname}"))


def _require_url_dsn(dsn: str) -> None:
    """Refuse a DSN this module would silently mangle.

    Keyword form (`dbname=effective host=…`) and the socket URI form (`postgresql:///effective`)
    both survive `urlsplit` and come out the other side as nonsense — `TEMPLATE` becomes
    `"dbname=effective host=localhost_template"`, the admin DSN becomes `"/postgres"` — and every
    one of those lands in a swallowed `except`, degrading to a full silent skip. Fail at import
    instead, where the message can name the culprit."""
    parts = urlsplit(dsn)
    if not parts.scheme.startswith("postgres") or not parts.netloc or not _dbname(dsn):
        raise pytest.UsageError(
            f"DATABASE_URL must be a `postgresql://host/dbname` URL; got {dsn.split('@')[-1]!r}. "
            "Keyword and socket-URI forms are mangled by the per-worker database naming."
        )


# xdist sets this in the worker process at `xdist/remote.py:417`, BEFORE `_prepareconfig` loads
# conftest — so the rewrite lands before any test module is imported. That timing is the whole
# trick: four other modules bind a DSN at import (`_durable.py`, `_conformance.py`,
# `test_absurd_integration.py`, `test_lineage_marker.py`). Publishing the worker's database
# through the ENVIRONMENT fixes all of them without touching any, and fixes the next one somebody
# writes, which an enumerated edit would not.
_WORKER = os.environ.get("PYTEST_XDIST_WORKER")  # "gw0", …; unset serially and in the controller
if _WORKER:
    _require_url_dsn(_BASE_DSN)
    os.environ["DATABASE_URL"] = _named(_BASE_DSN, f"{_dbname(_BASE_DSN)}_{_WORKER}")

DSN = os.environ.get("DATABASE_URL", _DEFAULT_DSN)

TEMPLATE = f"{_dbname(_BASE_DSN)}_template"
"""The prepared database the worker copies are made from — built by `scripts/pgtest_up.sh` (and
`scripts/setup_absurd.sh`) immediately after it bootstraps `effective`, so it is a pristine copy
of exactly what a fresh bootstrap produces.

`CREATE DATABASE … TEMPLATE` refuses while ANY session is connected to the source, so the rule is
that nothing holds a connection to it *while workers are being made*. That is a sequencing
promise, not an absence: `_require_usable_template` connects to read the disposability marker and
the migration version, and does it from the controller while it holds the cluster lock, before any
copy is attempted. An earlier version of this docstring claimed nothing ever connects, three lines
above the function that does."""


@pytest.fixture
def sqlite_app():
    """Build `SqliteApp`s that are closed when the test ends, however it ends.

    A factory rather than a single app, because tests routinely want two stores (a base and a
    fork target, two generations) and because the path is part of what they are asserting.
    Use it wherever a test would have written `SqliteApp(...)` directly::

        def test_x(tmp_path, sqlite_app):
            app = sqlite_app(str(tmp_path / "run.db"))

    **Why this exists.** 68 connections per suite run were never closed, which surfaced as
    `ResourceWarning: unclosed database` under `just test` — and only there, because the
    warning fires when the garbage collector reaches the connection, so it needs the load and
    the timing of the parallel lane to show up at all. It had been in the output for weeks.
    A trailing `app.close()` is the obvious repair and the wrong one: it is skipped by every
    early return and every raise, which is exactly when a test is already failing and least
    wants a second, unrelated failure mode.

    `ExitStack` + `closing` rather than a context manager on `SqliteApp`: the class already has
    the `close()` the stdlib needs, and a durable engine should not grow a `with` protocol just
    to satisfy its own test suite.
    """
    with ExitStack() as stack:

        def make(path: str = ":memory:", **kwargs) -> SqliteApp:
            return stack.enter_context(closing(SqliteApp(path, **kwargs)))

        yield make


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request):
    """The cross-engine sweep: every test taking `backend` runs twice, once per engine.

    It lives in `conftest` because two modules share it (`test_conformance.py` and
    `test_carrier_audit.py`); importing it instead would make pytest's collection depend on an
    import side effect."""
    from _conformance import AbsurdBackend, SqliteBackend
    from _durable import pg_ready

    if request.param == "postgres":
        if not pg_ready():
            pytest.skip("no Postgres with Absurd + ledger (just pgt-up)")
        b = AbsurdBackend()
    else:
        b = SqliteBackend()
    yield b
    b.close()


def _is_disposable(conn) -> bool:
    """`_durable.is_disposable`, the gate on every row this module writes."""
    from _durable import is_disposable

    return is_disposable(conn)


def _disposable_connection():
    """A session-long connection to this process's database, or `None` when it is unreachable or
    carries no disposability marker. Every row this module writes goes through it."""
    import psycopg

    try:
        conn = psycopg.connect(DSN, connect_timeout=2, autocommit=True)
    except Exception:
        return None  # no Postgres: every backend-gated test skips through its own `pg_ready()`
    try:
        if _is_disposable(conn):
            return conn
    except Exception:
        pass  # no permission to read the catalog: treat it as a database we may not write
    conn.close()
    return None


def _own_shared_database(conn) -> None:
    """Refuse a second serial run on the same database.

    Two serial runs share the base database, so each one's `work_batch()` claims the other's runs
    and defers them, and each one's reset cancels the other's. Held on `conn` for the session;
    Postgres drops it if the process dies. The two-key form keeps it apart from the cluster lock a
    parallel run takes, which uses only worker databases and so does not contend with this one."""
    held = conn.execute(
        t"SELECT pg_try_advisory_lock({_LOCK_KEY}, hashtext(current_database()))"
    ).fetchone()
    if not held[0]:
        conn.close()
        raise pytest.UsageError(
            f"another serial test run is using {_dbname(DSN)!r}; two runs on one database take "
            "each other's tasks. Wait for it, or run in parallel (`-n`), which gives each worker "
            "its own database."
        )


def _clear_claimable_tasks(conn) -> int:
    """Reset what makes a session non-idempotent, returning the tasks cancelled.

    A leftover claimable task spends later tests' `work_batch` claims, and a test that folds the
    WHOLE ledger into a projection passes on a clean database and fails on the next run with a
    duplicate key. So: cancel non-terminal tasks, then truncate the ledger and any projection a
    tree declares in `[tool.effective.test].app_tables`. A
    marked database without the Absurd schema is a half-built fixture, refused by name."""
    import psycopg
    from _durable import cancel_leftover_runs

    try:
        cancelled = cancel_leftover_runs(conn)
    except psycopg.errors.UndefinedTable as missing:
        raise pytest.UsageError(
            f"{_dbname(DSN)!r} is marked disposable and has no Absurd queue ({missing}); "
            "run `just pgt-up` or `just db-setup`."
        ) from missing
    for table in ("ledger", *(configured("app_tables", "test") or ())):
        found = conn.execute(t"SELECT to_regclass({table})").fetchone()
        if found is not None and found[0] is not None:
            conn.execute(t"TRUNCATE {table:i} CASCADE")
    return cancelled


_SWEEP_CONN = None


@pytest.fixture(autouse=True)
def _cancel_leftover_runs(request):
    """Cancel every run a test leaves non-terminal on the `default` queue.

    A worker claims whatever is ready, one task per `work_batch()`, and defers a task its app
    never registered by 15 to 30 seconds, after which it is claimable again. A leftover therefore
    spends a batch of whichever later test claims it, and a test that drives its run with a single
    batch runs nothing of its own. Cancelled rather than deleted, so a failed test's runs stay on
    disk to inspect. A run that must outlive its test (a parked run a module fixture holds) needs
    a queue of its own.

    A lost connection in a serial run ends the session, since the connection held the database's
    lock; in a worker, whose database is its own, it ends the sweep with one warning."""
    global _SWEEP_CONN
    yield
    if _SWEEP_CONN is None:
        return
    import warnings

    import psycopg
    from _durable import cancel_leftover_runs

    try:
        cancel_leftover_runs(_SWEEP_CONN)
    except psycopg.OperationalError as lost:
        _SWEEP_CONN = None
        if not _WORKER:
            # Set as well as raised: an exit raised beside another fixture's teardown error is
            # grouped with it, and the session would run on without the lock.
            reason = f"lost the lock on {_dbname(DSN)!r} after {request.node.nodeid}: {lost}"
            request.session.shouldstop = reason
            pytest.exit(reason)
        warnings.warn(
            f"leftover runs are no longer cancelled after {request.node.nodeid}: {lost}",
            stacklevel=1,
        )


def _alembic_head() -> str:
    """The migration this tree expects a database to be at."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parent.parent
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    head = ScriptDirectory.from_config(cfg).get_current_head()
    if head is None:  # no revisions at all: an incomplete checkout, not a drifted template
        raise pytest.UsageError(f"no alembic head under {root / 'migrations'} — checkout intact?")
    return head


def _require_usable_template(admin) -> None:
    """Refuse to fan out onto a template that is missing, precious, or behind the migrations.

    Each of the three is a way for a parallel run to look green while proving nothing, which is
    the failure this repo pays for most. Checked from the CONTROLLER, once, while it holds the
    cluster lock — so the connection this opens on the template is closed again before any worker
    exists to copy it. That ordering is the whole reason the lock is taken first: `CREATE DATABASE
    … TEMPLATE` refuses while ANY session is connected to the source, and this function is itself
    such a session."""
    import psycopg

    found = admin.execute(t"SELECT 1 FROM pg_database WHERE datname = {TEMPLATE}").fetchone()
    if found is None:
        raise pytest.UsageError(
            f"parallel run needs the template database {TEMPLATE!r}, which does not exist — "
            "run `just pgt-up` (container) or `just db-setup` (system Postgres), or drop `-n`."
        )
    with psycopg.connect(_named(_BASE_DSN, TEMPLATE), connect_timeout=2) as conn:
        if not _is_disposable(conn):
            raise pytest.UsageError(
                f"{TEMPLATE!r} carries no `_pgtest_disposable` marker, so this is not a throwaway "
                "fixture and a parallel run would create and DROP databases beside it. Refusing."
            )
        # `to_regclass` FIRST. A template snapshotted before `alembic upgrade` ran has no
        # `alembic_version` table at all, and selecting from it raises `UndefinedTable` out of a
        # configure hook — an INTERNALERROR instead of the refusal this function exists to give.
        # Found by review: the three refusals were each checked in a way that kept the table.
        present = conn.execute("SELECT to_regclass('alembic_version')").fetchone()
        at = None
        if present is not None and present[0] is not None:
            row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
            at = row[0] if row is not None else None
        head = _alembic_head()
        if at != head:
            raise pytest.UsageError(
                f"{TEMPLATE!r} is at migration {at!r}, the tree is at {head!r}. "
                "`just migrate` does not reach the template — re-run `just pgt-up` / `just "
                "db-setup` so the workers do not test an old schema."
            )


def _admin_connection():
    """A connection to `postgres` on the same server, or a REASON it could not be had.

    Returns `(conn, None)` or `(None, reason)`. The distinction is the point. An earlier version
    collapsed every failure into "no Postgres, do nothing", which is right for CI and wrong for a
    managed database: a role that can reach its OWN database but not the `postgres` maintenance
    database is the ordinary shape of a managed Postgres, and there the shrug turned a verification
    run into one that skipped every gated test and exited 0. A run named "verify" must not verify
    nothing."""
    import psycopg

    try:
        return psycopg.connect(
            _named(_BASE_DSN, "postgres"), connect_timeout=2, autocommit=True
        ), None
    except Exception as admin_failure:
        try:
            psycopg.connect(_BASE_DSN, connect_timeout=2).close()
        except Exception:
            return None, "no Postgres"  # nothing is reachable: CI, `test-core`, a dead DSN
        raise pytest.UsageError(
            "cannot reach the `postgres` database to provision per-worker copies "
            f"({admin_failure}), but {_dbname(_BASE_DSN)!r} itself IS reachable. A parallel run "
            "here would skip every backend-gated test and exit 0. Run serially (`-n0`), or "
            "grant the role access."
        ) from admin_failure


# One lock per CLUSTER, not per database — advisory locks are cluster-scoped, so two runs against
# DIFFERENT databases on one server refuse each other too. Right for the pg-test container, which
# is the case that matters; worth knowing if two checkouts ever share one managed database.
_LOCK_KEY = 0x0EFFEC71
_LOCK_CONN = None
_PROVISIONED: list[str] = []


def _distributing(config) -> bool:
    """Whether xdist will actually fan out — xdist's own predicate (`xdist/plugin.py:299`).

    `bool(tx)` alone is not it: `--tx popen --tx popen` with no `-n` leaves `dist == "no"`, so
    pytest runs serially while `tx` is non-empty, and provisioning off `tx` created worker
    databases nobody would ever use."""
    return config.getoption("dist") != "no" and bool(config.getoption("tx"))


def _require_pg_if_asked() -> None:
    """`EFFECTIVE_REQUIRE_PG=1` turns every Postgres skip into a refusal to start.

    A lane that skips its own subject reports success for a run that never happened:
    `just pgt-test` is 266 passed / 2 skipped against the container and 127 passed /
    144 skipped with nothing listening, exit 0 both times. Counting skips afterwards
    was the first answer and it measured the wrong thing — the claim is *the Absurd arm
    executed*, not *few tests skipped*, and the count moves whenever the authority
    census grows a row. This asserts the claim itself, once, before collection.

    Six functions in this tree answer "is Postgres ready" (`_conformance.pg_ready`,
    `_durable.pg_ready`, and four private spellings), so a per-guard check would be a
    domain enumerated as spellings of one referent — the shape
    `wiki/concepts/enforcer-domain.md` is about. The session is the one place that is
    upstream of all six.
    """
    if os.environ.get("EFFECTIVE_REQUIRE_PG") not in ("1", "true", "yes"):
        return
    from _durable import DSN, pg_ready

    if not pg_ready():
        raise pytest.UsageError(
            f"EFFECTIVE_REQUIRE_PG is set and {DSN.rsplit('@', 1)[-1]} has no reachable "
            "Postgres with the Absurd schema and the ledger. Run `just pgt-up`, or unset "
            "the variable to let the durable tests skip."
        )


ROLES = ("unit", "spine", "journey", "adversarial", "conformance", "property")
"""The test roles. A marker outside them, a future `slow` say, says nothing about a pass."""


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """A test that declares no role is a `unit` test.

    A test that proves more than one seam in isolation declares which of `ROLES`, and every other
    test is marked `unit` here, so `-m unit` selects exactly the population whose overlap is
    waste."""
    for item in items:
        if not any(item.get_closest_marker(role) for role in ROLES):
            item.add_marker(pytest.mark.unit)


def pytest_configure(config) -> None:
    """Controller-only: give every xdist worker its own copy of the template.

    Here rather than in each worker because `numprocesses` is already resolved from `auto` by the
    time this runs (xdist does it in `pytest_cmdline_main`, which precedes `_do_configure`), so
    the controller knows the whole worker list and can make the copies in sequence. Workers still
    self-heal in `pytest_sessionstart` — the list is not complete, because a worker that dies is
    replaced by one with a HIGHER id than any name here."""
    global _LOCK_CONN
    _require_pg_if_asked()
    if hasattr(config, "workerinput") or not _distributing(config):
        return  # a worker, or a serial run: the shared database, reset at session start as before

    # The controller names the template and the admin connection off the same URL the workers do,
    # so it owes the same check — otherwise a mangled DSN reaches `_require_usable_template` as a
    # confusing "template does not exist" instead of the real complaint.
    _require_url_dsn(_BASE_DSN)
    admin, _reason = _admin_connection()
    if admin is None:
        return  # nothing reachable; every gated test skips through its own `pg_ready()`

    # ONE RUN AT A TIME PER CLUSTER. Two runs sharing a container each drop and recreate
    # `<db>_gw0`, and `DROP DATABASE … WITH (FORCE)` terminates the other run's connections
    # mid-test — measured, and it surfaces as assertion failures that read as substrate bugs
    # (12 and 17 of them) rather than as "someone else is running the suite". Held for the whole
    # session on this connection; Postgres drops it if the process dies.
    if not admin.execute(t"SELECT pg_try_advisory_lock({_LOCK_KEY})").fetchone()[0]:
        admin.close()
        raise pytest.UsageError(
            f"another test run already owns the cluster behind {_dbname(_BASE_DSN)!r}. Parallel "
            "runs drop and recreate each other's worker databases — wait for it, or point this "
            "run at a different server."
        )
    _LOCK_CONN = admin

    _require_usable_template(admin)
    base = _dbname(_BASE_DSN)
    # Reap FIRST, including copies this run will not recreate. Drop-then-create bounds the count
    # by the largest `-n` the cluster ever saw, not by this run's: after one `-n 40`, a later
    # `-n auto` leaves 30 stale copies of a possibly-older schema sitting on the tmpfs.
    #
    # `_gw%` DELIBERATELY, and it is narrower than `_ensure_worker_database` can name: xdist's ids
    # are `gw0…` for every ordinary invocation, but `--tx 'popen//id=alpha'` self-heals into
    # `<base>_alpha`, which this never reaps. Widening to `<base>_%` would sweep a developer's
    # `effective_dev` off the same cluster, and a reaper that deletes more than it made is a worse
    # bug than a stray database. Exotic `--tx` ids are the user's to clean up.
    stale = admin.execute(
        t"SELECT datname FROM pg_database WHERE datname LIKE {base + r'\_gw%'}"
    ).fetchall()
    for (name,) in stale:
        admin.execute(t"DROP DATABASE IF EXISTS {name:i} WITH (FORCE)")
    for i in range(len(config.option.tx)):
        name = f"{base}_gw{i}"
        _create_worker_database(admin, name)
        _PROVISIONED.append(name)


def _create_worker_database(admin, name: str) -> None:
    """One copy of the template.

    FILE_COPY over the default WAL_LOG: WAL_LOG journals every page it copies, which is how a
    first attempt at this PANICked the server with `could not write to file "pg_wal/xlogtemp":
    No space left on device` and left it unable to recover — the test container's PGDATA is a
    tmpfs. Same speed, measured."""
    admin.execute(t"DROP DATABASE IF EXISTS {name:i} WITH (FORCE)")
    admin.execute(t"CREATE DATABASE {name:i} TEMPLATE {TEMPLATE:i} STRATEGY FILE_COPY")


_CLEARED = 0


def pytest_sessionstart(session) -> None:
    """Make sure this process has the database its DSN names, then reset it if it is shared.

    A WORKER checks rather than assumes. `pytest_configure` provisioned `gw0..gwN-1`, but xdist
    replaces a dead worker with a MONOTONIC id (`execnet` `Group.allocate_id`), so the first crash
    at `-n 10` produces `gw10` — a name nobody created. Without this the replacement's `pg_ready()`
    simply returns False and every backend-gated test on it SKIPS, leaving the run green and
    quieter. The durable lane is exactly where a worker dies, so this is not a corner.

    A SERIAL run locks the shared database and resets it instead. Either keeps its connection for
    the per-test sweep. `pytest_sessionstart`, NOT
    `pytest_report_header`: the header hook is not called at all under `-q`, and every gate in
    this repo runs `-q` — so the first version of this silently did nothing in exactly the runs it
    was written for. (Found by counting rows before and after a run rather than by trusting the
    hook.)"""
    global _CLEARED, _SWEEP_CONN
    if _PROVISIONED:
        return  # parallel controller: it runs no tests, and the base database is nobody's
    if _WORKER:
        _ensure_worker_database()
    if (conn := _disposable_connection()) is None:
        return
    if not _WORKER:
        _own_shared_database(conn)
        _CLEARED = _clear_claimable_tasks(conn)
    _SWEEP_CONN = conn


def _ensure_worker_database() -> None:
    """Create this worker's database if the controller did not name it. Loud on failure."""
    admin, _reason = _admin_connection()
    if admin is None:
        return  # no Postgres: this worker's gated tests skip, same as every other process
    with admin:
        name = _dbname(DSN)
        found = admin.execute(t"SELECT 1 FROM pg_database WHERE datname = {name}").fetchone()
        if found is not None:
            return
        try:
            _create_worker_database(admin, name)
        except Exception as failure:
            raise pytest.UsageError(
                f"worker {_WORKER} has no database {name!r} and could not make one ({failure}). "
                "Its backend-gated tests would all skip and the run would still be green."
            ) from failure


_LEAKED_FROM_WORKERS: list[str] = []


def pytest_testnodedown(node, error) -> None:
    """Collect each xdist worker's unclosed-connection sites as it shuts down.

    This exists because the obvious version of the gate was **silent in the only lane that
    matters**. `pytest_sessionfinish` runs in every worker process, so raising there fails the
    worker and not the run: measured, a deliberate leak exited 4 serially and **0** under
    `-n auto` — and `just test`, `just check` and `just cov` are all `-n auto`. The controller's
    own session never opened those connections, so it had nothing to report.

    `workeroutput` is the supported channel back. A worker fills it below; the controller
    accumulates here and enforces in `pytest_sessionfinish`.
    """
    _LEAKED_FROM_WORKERS.extend(getattr(node, "workeroutput", {}).get("unclosed_sqlite", []))


def pytest_sessionfinish(session, exitstatus) -> None:
    """Release the cluster lock, and fail the run if any SQLite connection was left open.

    The lock: Postgres would release it on disconnect anyway; being explicit means a `--pdb`
    session or an in-process re-run does not hold it for the rest of the day.

    The connections: `EFFECTIVE_TRACK_CONNECTIONS` (set at import, above) makes
    `effective.sqlite.connect` record a creation site per connection and drop it on an explicit
    `close()`, so what survives is exactly the set nobody closed — with a file and a line, which
    is the part the `ResourceWarning` cannot give you. That warning fires whenever the garbage
    collector happens to reach the object, so it lands on an unrelated test and names the
    collection site rather than the leak; 68 of them rode in the output for weeks for that
    reason. Counted instead, it is one number, and the number is enforceable.

    Under xdist a worker hands its sites to the controller rather than failing (see
    `pytest_testnodedown`); only the controller can fail the run.

    Enforced even when the run already failed, and deliberately: a leak found while triaging
    another failure is still a leak, and skipping the check then is how it comes back.

    **It PRINTS and sets the exit status; it does not raise, and that distinction cost real
    time.** The first version raised `pytest.UsageError` here, which pytest reports *instead of*
    the session's own summary — so a run with both a leak and a failing test showed the leak and
    swallowed the traceback entirely. A gate that hides the failures you are debugging is worse
    than the leak it reports; this one has to compose with a red run, because a red run is
    exactly when someone is reading the output.
    """
    global _LOCK_CONN, _SWEEP_CONN
    for held in (_LOCK_CONN, _SWEEP_CONN):
        if held is not None:
            held.close()
    _LOCK_CONN = _SWEEP_CONN = None

    mine = list(unclosed_connections())
    if (output := getattr(session.config, "workeroutput", None)) is not None:
        output["unclosed_sqlite"] = mine  # a worker: report upward, do not fail here
        return

    leaked = Counter(mine + _LEAKED_FROM_WORKERS)
    if leaked:
        sites = "\n".join(f"  {n:4d}  {site}" for site, n in leaked.most_common())
        print(
            f"\nLEAK GATE: {sum(leaked.values())} SQLite connection(s) were never closed, from "
            f"{len(leaked)} site(s):\n{sites}\n"
            "Counts are summed across xdist workers, so a module-level leak reads as one per "
            "worker where the source has a single construction.\n"
            "Build them with the `sqlite_app` fixture, which closes whatever it made however "
            "the test ends. A trailing `app.close()` is skipped by every early return."
        )
        if exitstatus == 0:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_report_header(config) -> str | None:
    """Say what this process did to the databases — provisioning, reset, or neither.

    The two cannot currently co-occur (`pytest_sessionstart` returns early once `_PROVISIONED` is
    non-empty), so this reads as belt-and-braces. It is: an earlier version returned the
    provisioning line INSTEAD of the reset line, which is how the reset lost its only report while
    still running. Reporting both is what makes that unrepresentable rather than merely absent."""
    lines = []
    if _PROVISIONED:
        lines.append(f"{len(_PROVISIONED)} worker copies of {TEMPLATE}")
    if _CLEARED > 0:
        lines.append(f"cancelled {_CLEARED} leftover task(s)")
    return f"test db: {', '.join(lines)}" if lines else None
