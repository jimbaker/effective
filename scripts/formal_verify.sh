#!/usr/bin/env bash
# The slow formal gate, BARE-METAL fallback for devices without Podman. Runs exactly the
# checks in scripts/formal_checks.sh — the same list the sandboxed default
# (formal_verify_container.sh) runs — preceded by the apalache jar hash check so a tampered
# or drifted dist refuses to run.
#
# Prefer `just formal-verify` (sandboxed: --network=none, read-only spec mount, tmpfs
# workdir). This path executes the npm dependency graph and the JVM against your real
# filesystem with network access; it exists for devices that cannot run Podman.
set -eo pipefail
cd "$(dirname "$0")/.."
PIN=infra/formal/PIN.txt
QUINT="$PWD/infra/formal/node_modules/.bin/quint"
[ -x "$QUINT" ] || { echo "formal-verify: no pinned quint — run 'just formal-setup' first." >&2; exit 1; }

APALACHE=$(awk '$1=="apalache"{print $2}' $PIN)
JAR="$HOME/.quint/apalache-dist-$APALACHE/apalache/lib/apalache.jar"
[ -f "$JAR" ] || { echo "formal-verify: apalache dist missing — run 'just formal-setup' first." >&2; exit 1; }
EXPECT=$(awk '$1=="apalache.jar.sha256"{print $2}' $PIN)
ACTUAL=$(sha256sum "$JAR" | cut -d' ' -f1)
if [ "$EXPECT" != "$ACTUAL" ]; then
    echo "formal-verify: apalache.jar sha256 MISMATCH (expected $EXPECT, got $ACTUAL) — refusing." >&2
    exit 1
fi

# shellcheck source=scripts/formal_checks.sh
source scripts/formal_checks.sh
cd formal/quint

run_check() {
    echo "==> $*"
    local t0=$SECONDS
    "$QUINT" "$@"
    echo "    (done in $((SECONDS-t0))s)"
}

# A check that must FAIL. `quint verify` exits non-zero on a violation, so a ZERO exit here
# means the counterexample vanished — the guard rotted, and that is the loud failure.
run_expect_violation() {
    local file="$1" args="$2" what="$3"
    echo "==> [expect violation] $file $args — $what"
    # shellcheck disable=SC2086
    if "$QUINT" verify "$file" $args >/dev/null 2>&1; then
        echo "    GUARD FAILED: $what, but the check PASSED" >&2
        return 1
    fi
    echo "    violated, as it must"
}

# A tooth: copy the model, flip its design toggle to the known-buggy value, assert the
# checker still finds the counterexample.
run_tooth() {
    local file="$1" from="$2" to="$3" args="$4" what="$5"
    echo "==> [tooth] $file: $from -> $to — $what"
    local dir
    dir=$(mktemp -d)
    sed "s/$from/$to/" "$file" > "$dir/bug_$file"
    # shellcheck disable=SC2086
    if (cd "$dir" && "$QUINT" verify "bug_$file" $args) >/dev/null 2>&1; then
        echo "    TOOTH FAILED: $what, but the check PASSED" >&2
        rm -rf "$dir"
        return 1
    fi
    rm -rf "$dir"
    echo "    broke, as it must"
}

for c in "${FORMAL_CHECKS[@]}"; do
    # shellcheck disable=SC2086
    run_check $c
done
for v in "${FORMAL_EXPECT_VIOLATION[@]}"; do
    IFS='|' read -r file args what <<< "$v"
    run_expect_violation "$file" "$args" "$what"
done
for t in "${FORMAL_TEETH[@]}"; do
    IFS='|' read -r file from to args what <<< "$t"
    run_tooth "$file" "$from" "$to" "$args" "$what"
done
echo "formal-verify (host): ${#FORMAL_CHECKS[@]} checks + ${#FORMAL_EXPECT_VIOLATION[@]} expected violations + ${#FORMAL_TEETH[@]} teeth passed."
