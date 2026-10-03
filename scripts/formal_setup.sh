#!/usr/bin/env bash
# Install the pinned formal toolchain (infra/formal/PIN.txt) on this machine.
# Idempotent. Never installs anything globally: quint lands in
# infra/formal/node_modules via `npm ci --ignore-scripts` (lockfile-integrity
# enforced), apalache is fetched by quint into ~/.quint and then hash-verified
# against the PIN before it is trusted.
set -eo pipefail
cd "$(dirname "$0")/.."
PIN=infra/formal/PIN.txt

# --- prerequisites we do NOT install: node/npm and a JDK -------------------
if ! command -v npm >/dev/null 2>&1; then
    echo "formal-setup: npm not found on PATH." >&2
    echo "  nvm machines: 'source ~/.nvm/nvm.sh' first." >&2
    exit 1
fi
JAVA_MAJOR=$(java -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+)[.".].*/\1/')
if [ -z "$JAVA_MAJOR" ] || [ "$JAVA_MAJOR" -lt 17 ]; then
    echo "formal-setup: Apalache needs a JDK >= 17 (found: ${JAVA_MAJOR:-none})." >&2
    exit 1
fi

# --- quint, pinned via lockfile integrity ----------------------------------
echo "==> quint $(awk '$1=="quint"{print $2}' $PIN) via npm ci (lockfile sha512, scripts disabled)"
(cd infra/formal && npm ci --ignore-scripts --no-audit --no-fund)
QUINT="$PWD/infra/formal/node_modules/.bin/quint"
"$QUINT" --version

# --- apalache: let quint fetch its pinned dist, then verify the jar hash ----
APALACHE=$(awk '$1=="apalache"{print $2}' $PIN)
DIST="$HOME/.quint/apalache-dist-$APALACHE"
if [ ! -f "$DIST/apalache/lib/apalache.jar" ]; then
    echo "==> fetching apalache $APALACHE (via a minimal quint verify)"
    (cd formal/quint && "$QUINT" verify effective.qnt --invariant=ledgerNodup --max-steps=1 >/dev/null)
fi
EXPECT=$(awk '$1=="apalache.jar.sha256"{print $2}' $PIN)
ACTUAL=$(sha256sum "$DIST/apalache/lib/apalache.jar" | cut -d' ' -f1)
if [ "$EXPECT" != "$ACTUAL" ]; then
    echo "formal-setup: apalache.jar sha256 MISMATCH — refusing to trust it." >&2
    echo "  expected $EXPECT (infra/formal/PIN.txt)" >&2
    echo "  actual   $ACTUAL ($DIST)" >&2
    echo "  If you bumped quint/apalache deliberately, re-record the hash from" >&2
    echo "  two independent machines per the PIN.txt procedure." >&2
    exit 1
fi
echo "==> apalache $APALACHE jar hash OK ($ACTUAL)"

# --- lean: pinned elsewhere (formal/lean/lean-toolchain via elan) -----------
if [ -x "$HOME/.elan/bin/lake" ] || command -v lake >/dev/null 2>&1; then
    echo "==> lean: lake present ($(PATH="$HOME/.elan/bin:$PATH" lake --version | head -1))"
else
    echo "==> lean: elan/lake not found — install elan to run 'just formal' Lean half" >&2
fi
echo "formal-setup: done. Fast gate: 'just formal'; full checks: 'just formal-verify'."
