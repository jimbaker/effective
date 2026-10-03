#!/usr/bin/env bash
# Bootstrap the local Absurd substrate for `effective` (sandbox Postgres).
#
# Requires a running local Postgres. Idempotent: safe to re-run. Applies the
# *vendored*, pinned schema (infra/absurd/PIN.txt — absurd 0.5.0, matched to
# absurd-sdk 0.5.0), migrating an older install in place.
#
# Needs no superuser step: absurd 0.5.0 has no uuid-ossp dependency, so there is no
# `CREATE EXTENSION`. The role/database creation assumes rights over the local cluster.
#
#   bash scripts/setup_absurd.sh
set -eo pipefail

DB=effective
ROLE=effective
PW=effective
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SQL="${ROOT}/infra/absurd/absurd.sql"
DSN="postgresql://${ROLE}:${PW}@localhost:5432/${DB}"

echo "==> role ${ROLE}"
psql -d postgres -v ON_ERROR_STOP=1 <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${ROLE}') THEN
    CREATE ROLE ${ROLE} LOGIN PASSWORD '${PW}';
  END IF;
END \$\$;
-- CREATEDB is for the PARALLEL test lane, which makes one database per xdist worker from a
-- template. The role never had it — the database below is created by the invoking user with
-- `createdb -O`, so nothing noticed. It reads as already-true on the pg-test container, where
-- the image makes `effective` a superuser; that is the container's doing, not this script's.
ALTER ROLE ${ROLE} CREATEDB;
SQL

echo "==> database ${DB}"
if [[ "$(psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB}'")" != "1" ]]; then
  createdb -O "${ROLE}" "${DB}"
fi

echo "==> apply pinned absurd.sql as ${ROLE} (fresh install, or migrate in place)"
# absurd.sql uses bare CREATE FUNCTION, so it cannot be re-applied over an existing
# install. A present schema is therefore NOT the same question as an up-to-date one:
# checking only for presence is how a re-vendor reaches every fresh database and none
# of the ones we already run on. Compare versions and apply the vendored migrations.
applied=$(PGPASSWORD="${PW}" psql "${DSN}" -tAc \
  "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='absurd' AND p.proname='create_queue'")
if [[ "${applied}" == "0" ]]; then
  PGPASSWORD="${PW}" psql "${DSN}" -v ON_ERROR_STOP=1 -f "${SQL}" >/dev/null
  echo "    installed $(PGPASSWORD="${PW}" psql "${DSN}" -tAc 'SELECT absurd.get_schema_version()')"
else
  have=$(PGPASSWORD="${PW}" psql "${DSN}" -tAc "SELECT absurd.get_schema_version()")
  want=$(sed -n 's/^tag: //p' "${ROOT}/infra/absurd/PIN.txt")
  if [[ "${have}" == "${want}" ]]; then
    echo "    absurd schema already at ${have} — nothing to do"
  else
    mig="${ROOT}/infra/absurd/migrations/${have}-${want}.sql"
    if [[ -f "${mig}" ]]; then
      echo "    migrating ${have} -> ${want}"
      PGPASSWORD="${PW}" psql "${DSN}" -v ON_ERROR_STOP=1 -f "${mig}" >/dev/null
      now=$(PGPASSWORD="${PW}" psql "${DSN}" -tAc "SELECT absurd.get_schema_version()")
      [[ "${now}" == "${want}" ]] || { echo "    MIGRATION DID NOT TAKE: still ${now}" >&2; exit 1; }
    else
      echo "    no vendored migration ${have} -> ${want} (looked for ${mig})" >&2
      exit 1
    fi
  fi
fi

echo "==> queue 'default'"
has=$(PGPASSWORD="${PW}" psql "${DSN}" -tAc "SELECT count(*) FROM absurd.list_queues() WHERE queue_name='default'")
if [[ "$has" != "1" ]]; then
  PGPASSWORD="${PW}" psql "${DSN}" -v ON_ERROR_STOP=1 -c "SELECT absurd.create_queue('default');" >/dev/null
fi

echo "==> apply our schema via alembic (ledger)"
( cd "${ROOT}" && DATABASE_URL="${DSN}" uv run alembic upgrade head )

# The TEMPLATE the parallel test lane copies, taken from `${DB}` the moment it is up to date.
# Refreshed on every run because `just migrate` reaches `${DB}` and not the template, and a
# template behind the migrations is a parallel run testing an old schema. `tests/conftest.py`
# checks `alembic_version` against the tree and refuses rather than letting that pass.
#
# Unlike the pg-test container this database is NOT marked disposable, so conftest will refuse to
# fan out onto it — deliberately. A dev database is not a fixture, and the parallel lane creates
# and drops databases beside its template.
#
# BUILD FIRST, SWAP AFTER — the order is the whole point. `CREATE DATABASE … TEMPLATE` requires
# ZERO other sessions on the source, so `just serve`, the drain, the Shiny board or an open psql
# all make it fail, and this script's header promises it is idempotent and safe to re-run.
# Dropping the old template first and only then discovering the source is busy leaves the cluster
# with NO template, which is worse than leaving a stale one. Checking `pg_stat_activity` first
# only narrows that window — a session arriving between the check and the CREATE still loses it
# (reproduced by review). Creating under a scratch name is order-independent: if the CREATE
# fails, nothing has been destroyed and the existing template is untouched.
echo "==> template ${DB}_template (per-worker copies for -n auto)"
ADMIN="${DSN%/*}/postgres"
PGPASSWORD="${PW}" psql "${ADMIN}" -v ON_ERROR_STOP=1 -q \
  -c "DROP DATABASE IF EXISTS ${DB}_template_new WITH (FORCE);"
if PGPASSWORD="${PW}" psql "${ADMIN}" -v ON_ERROR_STOP=1 -q \
     -c "CREATE DATABASE ${DB}_template_new TEMPLATE ${DB};" 2>/dev/null; then
  PGPASSWORD="${PW}" psql "${ADMIN}" -v ON_ERROR_STOP=1 -q \
    -c "DROP DATABASE IF EXISTS ${DB}_template WITH (FORCE);" \
    -c "ALTER DATABASE ${DB}_template_new RENAME TO ${DB}_template;"
else
  PGPASSWORD="${PW}" psql "${ADMIN}" -q -c "DROP DATABASE IF EXISTS ${DB}_template_new;" || true
  echo "    ${DB} is in use — template left as it was (close other sessions and re-run"
  echo "    if you need the parallel test lane, \`just test\`'s -n auto)."
fi

echo "Absurd substrate ready: ${DSN}"
