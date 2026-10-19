#!/usr/bin/env bash
# Stand up Odoo Community and its Postgres in one rootless Podman pod, with the `spike` database
# initialized with accounting and no demo data. Loopback only; the database lives on tmpfs, so
# every `odoo-up` starts clean.
#
#   bash scripts/odoo_up.sh
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POD=effective-odoo
PORT="${ODOO_PORT:-18069}"
pinned() { echo "$(sed -n 's/^image:[[:space:]]*//p' "$1")@$(sed -n 's/^digest:[[:space:]]*//p' "$1")"; }
ODOO="$(pinned "${ROOT}/infra/odoo/PIN.txt")"
PG="$(pinned "${ROOT}/infra/pg-test/PIN.txt")"
CONNECT=(-e HOST=127.0.0.1 -e USER=odoo -e PASSWORD=odoo)

podman pod rm -f "${POD}" >/dev/null 2>&1 || true
podman pod create --name "${POD}" -p "127.0.0.1:${PORT}:8069" >/dev/null
podman run -d --pod "${POD}" --name "${POD}-db" \
  -e POSTGRES_USER=odoo -e POSTGRES_PASSWORD=odoo -e POSTGRES_DB=postgres \
  --health-cmd 'pg_isready -U odoo' --health-interval 1s --health-retries 30 \
  --tmpfs /var/lib/postgresql "${PG}" -c fsync=off >/dev/null
podman wait --condition=healthy "${POD}-db" >/dev/null

echo "==> initialize the spike database with accounting"
podman run --rm --pod "${POD}" "${CONNECT[@]}" "${ODOO}" \
  odoo -d spike -i account --stop-after-init >/dev/null 2>&1

# Odoo 20 binds 127.0.0.1 inside the container unless told otherwise, which the pod's port
# forward cannot reach. Worker processes serve concurrent tests in parallel; one threaded process
# serves them in turn.
podman run -d --pod "${POD}" --name "${POD}-app" "${CONNECT[@]}" "${ODOO}" \
  odoo -d spike --db-filter='^spike$' --http-interface=0.0.0.0 \
  --workers="${ODOO_WORKERS:-6}" --max-cron-threads=0 >/dev/null
echo "==> wait for Odoo on ${PORT}"
curl -sf -o /dev/null --retry 90 --retry-delay 1 --retry-all-errors "http://127.0.0.1:${PORT}/web/login"
echo "==> ready"
