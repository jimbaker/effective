#!/usr/bin/env bash
# The fast Quint spec gate, sandboxed: typecheck every model inside the
# effective-formal container with NO network, a read-only spec mount, and a
# tmpfs workdir. quint's ~79-package npm graph never executes with network
# access or a view of the host filesystem — the same first-line supply-chain
# defense as the (already-containerized) `formal-verify` slow gate.
#
# Image: build once per device with `just formal-image`.
# Host fallback (no Podman): `just formal-quint-host`.
set -eo pipefail
cd "$(dirname "$0")/.."
IMG=effective-formal:0.32.0-0.56.1

podman image exists "$IMG" || {
    echo "formal-quint (container): image $IMG not built — run:" >&2
    echo "  just formal-image" >&2
    exit 1
}

SPECS=(formal/quint/*.qnt)

for path in "${SPECS[@]}"; do
    spec=${path##*/}
    echo "==> quint typecheck $spec"
    podman run --rm --network=none \
        -v ./formal/quint:/spec:ro,Z \
        --tmpfs /work:rw,size=512m \
        "$IMG" typecheck "$spec"
done
echo "formal-quint (container): all ${#SPECS[@]} specs typecheck."
