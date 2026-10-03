"""What the gate needs from the WORLD — probed once, reported by preflight, read at the call sites.

`preflight.py` checks the gate's TOOLCHAIN: which unlocked commands `just check` reaches and
whether each pins its version. This is the other axis — what those commands need from the
environment they run in: money to spend, a service to talk to, a container runtime to start.

**Absence is normal and is never fatal here**. A sandbox with no egress, a
laptop with the database down, a machine with no API key: each is a good reason, and a gate that
refuses to run at all teaches nothing about the 4000 tests that would have passed. So every probe
below REPORTS, and `just check` continues. What absence costs is named, and so is the one command
that buys it back.

**Two kinds of precondition, and only one is probeable.** A *configured* precondition — a key on
disk, a port answering, an image built — can be checked before anything runs, which is what these
probes do. **API AVAILABILITY cannot**: a provider that is up when preflight asks and rate-limits
ninety seconds later is indistinguishable, in advance, from one that is up. It is discovered by
the attempt failing, so it belongs to the call site rather than to preflight, and the call site's
obligation is the same one — be loud, and continue. Measured 2026-08-27: two `tape-check` runs
minutes apart aborted at 0 steps against a valid credential, and two later runs banked cleanly.
The gate went red for a provider hiccup.

**The capability -> recipe mapping below is DECLARED, and this docstring owes the reader why**,
because `preflight.py` derives its own domain and says a hand-written list is the failure mode this
repo keeps re-learning. There is no cheap derivation from "recipe" to "needs a credential": the
requirement lives several imports deep (a recipe's script -> an agent engine -> the model client),
so deriving it means an import-closure walk of the kind `lint._effect_module_closure` already does
for effect modules. That is the principled version and it is not built. Until it is, a new gate
recipe with a new requirement will be one entry short here — exactly the shape being warned about,
named rather than hidden.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective"
)


@dataclass(frozen=True)
class Precondition:
    """One capability the gate would like, and what its absence costs.

    `covers` is what goes UNRUN without it — the sentence that keeps a green gate honest about its
    own domain, which is this repo's recurring lesson in the form it keeps recurring: a measurement
    licenses a claim only over what it covered.
    """

    name: str
    available: bool
    detail: str
    enable: str
    covers: str
    warning: str = ""
    """A hazard that does NOT make the capability absent — reported for an available one too.

    Presence is not validity, and the gap between them is where this repo has actually lost time:
    a credential can be configured, found, and wrong. `available` answers "is something there";
    this answers "is what is there going to bite you"."""


def _api_env_key(name: str = "OPENAI_API_KEY") -> str | None:
    """The value `./api.env` declares for `name`, or None. Never logged — only its LENGTH is,
    which is what made the conflict below diagnosable and is not the secret."""
    f = REPO / "api.env"
    if not f.exists():
        return None
    for raw in f.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().removeprefix("export ").strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def spend() -> Precondition:
    """A credential the live paths can reach: presence, plus the ONE validity-ish check that is
    free, a conflict between two sources of the key.

    Both env and `api.env` count, because the agent engine's `load_api_env` loads the latter when
    the former is unset (the loader walks to the repo root and finds it).

    **The conflict arm is the reason this is not three lines.** `load_api_env` returns EARLY when
    the environment already carries a key, so an environment key WINS — and `justfile`'s
    `set dotenv-load` searches PARENT directories, so a `.env` outside the repo silently supplies
    one to every recipe. Measured 2026-08-27: a `.env` in a parent directory put a 56-character key
    into every `just` recipe while `./api.env` held the working 164-character one, and
    `just tape-check` therefore aborted its live run at 0 steps. The gate then reported *"the fresh
    tape does not replay: model called during replay"*, which is the empty tape talking and reads
    like a broken loop. Two runs bypassing `just` succeeded, which is what made it look transient.

    A presence check cannot see this, and a validity check would need a paid round trip. Comparing
    the two configured credentials costs nothing and names the exact hazard, so it is worth more
    here than either.
    """
    env_key = os.environ.get("OPENAI_API_KEY")
    file_key = _api_env_key()
    if not env_key and not file_key:
        return Precondition(
            "spend",
            False,
            "no OPENAI_API_KEY in env or ./api.env",
            "put OPENAI_API_KEY in ./api.env, or export it",
            "tape-check cannot bank a fresh tape; the banked-tape lane skips",
        )
    if env_key and file_key and env_key != file_key:
        return Precondition(
            "spend",
            True,
            f"OPENAI_API_KEY in env (len {len(env_key)}) — WINS over ./api.env (len "
            f"{len(file_key)})",
            "",
            "",
            warning=(
                "two DIFFERENT credentials are configured and the environment one wins wherever "
                "a caller checks presence and stops. If live runs abort at 0 steps, this is why, "
                "and note `just`'s `set dotenv-load` searches PARENT directories, so the env one "
                "may come from a `.env` outside this repo. Fix: `unset OPENAI_API_KEY`, or give "
                "the SCRIPT a `load_api_env` that overrides, as a live bank script should. "
                "A gate recipe cannot source it: `preflight`'s own ungoverned-tool rule reports "
                "every command line that is not `uv run`, shell builtins included."
            ),
        )
    where = "OPENAI_API_KEY in env" if env_key else "OPENAI_API_KEY in ./api.env"
    return Precondition("spend", True, where, "", "")


def postgres() -> Precondition:
    """The second durable engine. Without it the whole Absurd half of the cross-engine sweep is
    absent, and a green run means less than it looks — the case CLAUDE.md records at 168 tests.

    The same predicate `tests/_durable.pg_ready` uses. One definition with two readers is the
    point; a second spelling of "is Postgres ready" is how the two drift.
    """
    try:
        import psycopg
    except ImportError:  # pragma: no cover - psycopg is a hard dep of the durable lane
        return Precondition(
            "postgres", False, "psycopg not importable", "uv sync", "the Absurd conformance half"
        )
    try:
        with psycopg.connect(PG_DSN, connect_timeout=2) as conn:
            # `LIMIT 0` returns NO ROWS. This is a schema reachability probe — does the table
            # exist and can we speak to it — not a fold, so there is nothing for `hypothetical`
            # to filter, and adding the predicate would advertise a canonical read this is not.
            # Same query as `tests/_durable.pg_ready`; the gate does not scan that one because
            # `tests` is deliberately out of its domain, so this is its first scanned copy.
            # lint: ledger-read-not-canonical: reachability probe, LIMIT 0, reads no rows
            conn.execute("SELECT 1 FROM ledger LIMIT 0")
            conn.execute("SELECT 1 FROM pg_namespace WHERE nspname='absurd'")
    except Exception as exc:
        return Precondition(
            "postgres",
            False,
            f"{type(exc).__name__} against the configured DSN",
            "just pgt-up  (its PGDATA is tmpfs — it does not survive a podman restart)",
            "the [postgres] half of the conformance sweep; SQLite-only is not a port",
        )
    return Precondition("postgres", True, "reachable, absurd schema + ledger present", "", "")


def container() -> Precondition:
    """A rootless container runtime. The pinned images (`code-agent`, `pg-test`, `elk`, `formal`)
    all rest on it, and `tape-check` needs the code-agent one to bank."""
    if (podman := shutil.which("podman")) is None:
        return Precondition(
            "container",
            False,
            "podman not on PATH",
            "install podman (rootless)",
            "tape banking, pgt-up, the ELK and formal lanes",
        )
    try:
        subprocess.run([podman, "info"], capture_output=True, timeout=15, check=True)
    except Exception as exc:
        return Precondition(
            "container",
            False,
            f"podman present but not usable ({type(exc).__name__})",
            "check the rootless setup: podman info",
            "tape banking, pgt-up, the ELK and formal lanes",
        )
    return Precondition("container", True, "podman usable", "", "")


def probe_all() -> list[Precondition]:
    """Every configured precondition, in the order a reader cares about them."""
    return [spend(), postgres(), container()]


DISCOVERED_AT_USE = (
    "api-availability: not probeable in advance — a provider up now can rate-limit later. "
    "A failed live bank is reported and the gate continues (by the bank script)."
)
"""The fourth precondition, which has no probe and says so rather than pretending."""
