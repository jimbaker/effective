#!/usr/bin/env bash
# Copy the read-only spec mount into the writable workdir, then run quint.
set -eo pipefail
if [ -d /spec ]; then
    cp /spec/*.qnt /work/ 2>/dev/null || true
fi
exec quint "$@"
