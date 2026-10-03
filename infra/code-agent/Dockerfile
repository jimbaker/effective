# The T-real container tier: the sandbox the coding machine runs a model-written tree in.
#
# A fully-capable real bash + real git + pytest — the success predicate runs here
# with a real toolchain (no exit-127 interpreter gaps). The working tree is
# rebuilt into /work per op by a base64-on-stdin materialize script (NO volume
# mount, NO overlayfs — a fresh `podman run --rm` container each time), and real
# git provides commits/history. Per the two-bookkeepers invariant the container's
# git tree is *disposable execution state* and the ledger stays canonical (git SHA
# is a projection).
#
# Supply-chain policy (CLAUDE.md): the base image is pinned by DIGEST (not a
# floating tag); pytest is pinned exact. See PIN.txt for the recorded digests and
# the cross-machine corroboration line. Built + run rootless under podman.
FROM docker.io/library/python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --root-user-action=ignore pytest==8.3.4

# Deterministic commit identity: a fixed author/committer (dates are pinned per-commit
# via GIT_AUTHOR_DATE/GIT_COMMITTER_DATE at commit time) so a clean re-run reproduces the
# same git SHA — the ledger's recorded projection is reproducible.
RUN git config --global user.email agent@effective.local \
 && git config --global user.name "effective code agent" \
 && git config --global init.defaultBranch main \
 && git config --global advice.detachedHead false

WORKDIR /work
