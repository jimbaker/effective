#!/usr/bin/env bash
# Stand up the stateful durable-replay test Postgres under rootless Podman.
#
# A real, multi-connection Postgres pinned BY DIGEST (infra/pg-test/PIN.txt) —
# the supply-chain-minimal store for the crash-at-every-op durability tests. Daemonless,
# rootless, loopback-only. Idempotent: re-run to (re)bootstrap a clean DB.
#
#   bash scripts/pgtest_up.sh
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=effective-pg-test
PORT="${PGTEST_PORT:-5432}"
DSN="postgresql://effective:effective@localhost:${PORT}/effective"
DB_TEMPLATE=effective_template

# Pin: reference the image ONLY by digest (never the mutable tag).
IMAGE="$(sed -n 's/^image:[[:space:]]*//p' "${ROOT}/infra/pg-test/PIN.txt")"
DIGEST="$(sed -n 's/^digest:[[:space:]]*//p' "${ROOT}/infra/pg-test/PIN.txt")"
REF="${IMAGE}@${DIGEST}"

# RAM-BACKED and fsync-free. Postgres has no `:memory:` mode, but a tmpfs data directory plus
# `fsync=off` is the practical equivalent, and both knobs were measured here (2026-07-25,
# `just pgt-test`, 127 tests):
#
#   disk + fsync on   15.5s   <- Postgres's default
#   disk + fsync off  12.5s
#   tmpfs + fsync off 12.5s
#
# So the ~20% is entirely `fsync=off`; tmpfs adds no speed, because the page cache was already
# absorbing the writes. tmpfs earns its place on WEAR instead: a fresh cluster is 47 MB and a run
# churns ~16 MB of WAL, and on tmpfs none of that ever reaches flash. Minor per run, free to
# eliminate, and this container is recreated on every `pgt-up` anyway.
#
# `fsync=off` does NOT weaken what these tests prove. The durability under test is the SUBSTRATE's
# replay semantics — checkpoints committed through SQL, worker death simulated in Python — not
# Postgres's own crash safety. No test kills the postgres process. (If one ever does, it must set
# fsync back on for that case and say why.)
#
# SIZED FOR THE PARALLEL LANE, not for one database. `-n auto` gives each xdist worker its own
# copy of the template (see `tests/conftest.py`), so the cluster holds ~20 databases, plus the WAL
# for creating them, plus whatever they grow to during a run. The old 1g default was not enough:
# copying a bloated 32 MB database 20 ways PANICked the server with `could not write to file
# "pg_wal/xlogtemp": No space left on device` and left it unable to restart. A tmpfs only occupies
# the RAM it actually uses, so the ceiling costs nothing until it is needed.
#
# PGDATA is a SUBDIRECTORY of the tmpfs mount, not the mount itself: the image refuses a mount
# point as PGDATA, and initdb needs to create the directory itself to get 0700 on it.
echo "==> (re)create container ${NAME} from ${REF} (tmpfs PGDATA, fsync off)"
podman rm -f "${NAME}" >/dev/null 2>&1 || true
podman run -d --name "${NAME}" \
  -e POSTGRES_USER=effective -e POSTGRES_PASSWORD=effective -e POSTGRES_DB=effective \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  --tmpfs "/var/lib/postgresql/data:rw,size=${PGTEST_TMPFS_SIZE:-4g},mode=0777" \
  -p "127.0.0.1:${PORT}:5432" \
  "${REF}" -c fsync=off -c synchronous_commit=off -c full_page_writes=off >/dev/null

echo -n "==> wait for readiness"
ready=0
for _ in $(seq 1 30); do
  if podman exec "${NAME}" pg_isready -U effective -d effective -q 2>/dev/null; then
    echo " — ready"; ready=1; break
  fi
  echo -n "."; sleep 1
done
# Without this, falling through the loop would continue, and the next `psql` would fail with a
# connection error that leaves the reader to work backwards to "it never came up".
[ "${ready}" = 1 ] || { echo; echo "${NAME} never became ready in 30s" >&2; exit 1; }

# The container is fresh every time, so absurd.sql applies from scratch — no migration
# path is needed here, unlike setup_absurd.sh. 0.5.0 needs no uuid-ossp extension.
echo "==> vendored absurd.sql (pinned — see infra/absurd/PIN.txt) + queue 'default'"
podman exec -i "${NAME}" psql -U effective -d effective -v ON_ERROR_STOP=1 -q -f - < "${ROOT}/infra/absurd/absurd.sql"
podman exec "${NAME}" psql -U effective -d effective -v ON_ERROR_STOP=1 -tAc "SELECT absurd.create_queue('default');" >/dev/null

# The DISPOSABILITY MARKER. A test session resets more than tasks — it truncates the ledger and
# the projections, because a global rebuild folds the whole ledger and a leftover row from a
# previous session collides. That reset must never fire on somebody's dev database, and
# DATABASE_URL alone cannot tell them apart (both are `effective` on 5432). So the database
# DECLARES itself disposable here, and `tests/conftest.py` / `scripts/pgtest_clean.py` refuse the
# destructive half without it.
podman exec "${NAME}" psql -U effective -d effective -v ON_ERROR_STOP=1 -q \
  -c 'CREATE TABLE IF NOT EXISTS _pgtest_disposable (created_at timestamptz DEFAULT now());' 

echo "==> our schema via alembic (ledger + trigger, projections)"
( cd "${ROOT}" && DATABASE_URL="${DSN}" uv run alembic upgrade head >/dev/null )

# The TEMPLATE the parallel lane copies. Taken here, from `effective` itself the moment it is
# bootstrapped, so the template is by construction a pristine copy of exactly what a fresh
# bootstrap produces — including the disposability marker, which is what lets `tests/conftest.py`
# tell a throwaway fixture from somebody's dev database before it starts creating and dropping
# databases beside it. Nothing ever connects to the template afterwards, which is what keeps
# `CREATE DATABASE … TEMPLATE` from tripping "source database is being accessed by other users".
echo "==> template ${DB_TEMPLATE} (per-worker copies for -n auto)"
podman exec "${NAME}" psql -U effective -d postgres -v ON_ERROR_STOP=1 -q \
  -c "DROP DATABASE IF EXISTS ${DB_TEMPLATE} WITH (FORCE);" \
  -c "CREATE DATABASE ${DB_TEMPLATE} TEMPLATE effective;"

echo "pg-test ready: ${DSN}"
