#!/usr/bin/env bash
# The slow formal gate, sandboxed (the DEFAULT): the registered Quint checks run inside
# the effective-formal container with NO network, a read-only spec mount, and a
# tmpfs workdir — the npm dependency graph and the fetched Apalache/TLC jars
# never execute with network access or a view of the host filesystem.
#
# Image: build once per device with
#   podman build -t effective-formal:0.32.0-0.56.1 infra/formal/
# The build itself enforces the supply chain: npm ci --ignore-scripts against
# the lockfile, and the Apalache jar sha256-checked against infra/formal/PIN.txt
# (the build FAILS on mismatch, so a running image implies a verified jar).
set -eo pipefail
cd "$(dirname "$0")/.."
IMG=effective-formal:0.32.0-0.56.1

# The check LIST is shared with the host fallback (scripts/formal_verify.sh) so the two gates
# run the same checks.
# shellcheck source=scripts/formal_checks.sh
source scripts/formal_checks.sh

podman image exists "$IMG" || {
    echo "formal-verify-container: image $IMG not built — run:" >&2
    echo "  podman build -t $IMG infra/formal/" >&2
    exit 1
}

run() {
    echo "==> quint $*"
    local t0=$SECONDS
    podman run --rm --network=none \
        -v ./formal/quint:/spec:ro,Z \
        --tmpfs /work:rw,size=512m \
        "$IMG" "$@"
    echo "    (done in $((SECONDS-t0))s)"
}

run_check() { run "$@"; }

# A check that must FAIL: `quint verify` exits non-zero on a violation, so a ZERO exit means
# the counterexample vanished — the guard rotted, and that is the loud failure.
run_expect_violation() {
    local file="$1" args="$2" what="$3"
    echo "==> [expect violation] $file $args — $what"
    # shellcheck disable=SC2086
    if podman run --rm --network=none -v ./formal/quint:/spec:ro,Z --tmpfs /work:rw,size=512m \
           "$IMG" verify "$file" $args >/dev/null 2>&1; then
        echo "    GUARD FAILED: $what, but the check PASSED" >&2
        exit 1
    fi
    echo "    violated, as it must"
}

# A tooth: copy the model, flip its design toggle to the known-buggy value, assert the checker
# still finds the counterexample.
run_tooth() {
    local file="$1" from="$2" to="$3" args="$4" what="$5"
    echo "==> [tooth] $file: $from -> $to — $what"
    local dir
    dir=$(mktemp -d)
    chmod 755 "$dir"
    sed "s/$from/$to/" "formal/quint/$file" > "$dir/bug_$file"
    chmod 644 "$dir/bug_$file"
    # shellcheck disable=SC2086
    if podman run --rm --network=none -v "$dir":/spec:ro,Z --tmpfs /work:rw,size=512m \
           "$IMG" verify "bug_$file" $args >/dev/null 2>&1; then
        echo "    TOOTH FAILED: $what, but the check PASSED" >&2
        rm -rf "$dir"; exit 1
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
echo "formal-verify (container): ${#FORMAL_CHECKS[@]} checks + ${#FORMAL_EXPECT_VIOLATION[@]} expected violations + ${#FORMAL_TEETH[@]} teeth passed."
