#!/usr/bin/env bash
# Stand up the arrivals-bridge test Redis under rootless Podman.
#
# A real Redis pinned BY DIGEST (infra/redis-test/PIN.txt), for the Redis Streams
# delivery semantics the bridge's recovery contract rests on. Daemonless, rootless,
# loopback-only. Idempotent: re-run for a clean, empty server.
#
#   bash scripts/redistest_up.sh
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=effective-redis-test
PORT="${REDISTEST_PORT:-6379}"

# Pin: reference the image ONLY by digest (never the mutable tag).
IMAGE="$(sed -n 's/^image:[[:space:]]*//p' "${ROOT}/infra/redis-test/PIN.txt")"
DIGEST="$(sed -n 's/^digest:[[:space:]]*//p' "${ROOT}/infra/redis-test/PIN.txt")"
REF="${IMAGE}@${DIGEST}"

# NO PERSISTENCE, deliberately, for correctness rather than speed. A stream's retention and its
# consumer groups are what the bridge's recovery contract rests on, so every test that cares
# about redelivery must build the state it reads inside its own run. An RDB or AOF file
# surviving between runs would let a test pass on state some earlier run left, which is the
# shape of green that means nothing. `--save ''` turns off RDB snapshots and `--appendonly no`
# the AOF, so a restart is always an empty server.
#
# The bridge itself is a different question: a DEPLOYED bridge wants persistence, because a
# stream that loses its entries loses the evidence a decision was made from. That belongs in
# the bridge's ADR, not here.
echo "==> (re)create container ${NAME} from ${REF} (no persistence)"
podman rm -f "${NAME}" >/dev/null 2>&1 || true
podman run -d --name "${NAME}" \
  -p "127.0.0.1:${PORT}:6379" \
  "${REF}" redis-server --save '' --appendonly no >/dev/null

echo -n "==> wait for readiness"
ready=0
for _ in $(seq 1 30); do
  if podman exec "${NAME}" redis-cli ping 2>/dev/null | grep -q PONG; then
    echo " — ready"; ready=1; break
  fi
  echo -n "."; sleep 1
done
# Falling through the loop and continuing would make the next command fail with a connection
# error, leaving the reader to work backwards to "it never came up".
[ "${ready}" = 1 ] || { echo; echo "${NAME} never became ready in 30s" >&2; exit 1; }

echo "redis-test ready: redis://localhost:${PORT}/0"
