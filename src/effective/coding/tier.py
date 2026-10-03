"""Where a coding machine's tools actually EXECUTE — the host, or a pinned container.

The machine's success predicate runs `pytest` over a tree a model wrote. That is arbitrary code
execution by design: pytest imports every module it collects, and a `conftest.py` in the tree runs
at collection time, before a single test does. On the host it runs as the user driving the machine,
with that user's filesystem, network and credentials.

**The tier is a seam rather than a setting**, in the sense this project uses the word: a named,
typed, swappable point with a measurable cost. `run_suite`'s contract already anticipated it — *"a
deployment may substitute a different command; what it may not do is return a judgment instead of
a measurement"* — and this is that substitution made explicit, so a caller declares where it wants
execution to happen instead of discovering that the answer was always "here".

**What is containerized, and what deliberately is not.** Only execution. `effective.coding.gate`
shells out to ruff, and `coding.edits.structural` to ast-grep and jedi, and none of the three RUNS
the content: they parse it. Containerizing a parser would buy no isolation and would cost the gate
its access to this repo's ruff configuration, whose absence the gate treats as a refusal rather
than a pass. Naming the distinction is the point — a blanket "everything in a container" reads as
more careful and is mostly ceremony.

**No volume mount, no overlayfs**, matching the discipline `infra/code-agent/Dockerfile` states
for the same image: the tree is unpacked inside a fresh `--rm` container from a tar fed on stdin,
so file content needs no host tempfile and nothing of the host is reachable from inside.
With `--network=none` there is no egress either. A container that cannot see the host filesystem
and cannot reach the network is the whole of what this buys, and it is bought per op.

**The default stays HOST, which is a deliberate choice and not an oversight.** Every existing
caller — the conformance suite, the bench, the machine's own tests — runs without podman, and
silently requiring an image would turn a missing container into a mass of failures that look like
logic errors. A surface that drives a model to write code chooses `CONTAINER` explicitly; the TUI
does. `Tier.CONTAINER` with no image REFUSES loudly, because absence must not read as isolation.
"""

import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import cache
from pathlib import PurePosixPath

from effective.ops import Unretryable

IMAGE = "effective-code-agent:pin"
"""The pinned image, built from `infra/code-agent/` with
`podman build -q -t effective-code-agent:pin -f infra/code-agent/Dockerfile infra/code-agent`.

It bakes a real bash, a real git and a pinned pytest on a digest-pinned base, and
`infra/code-agent/PIN.txt` is the supply-chain record."""

WORKDIR = "/work"
TIER_ENV = "EFFECTIVE_EXEC_TIER"


class Tier(StrEnum):
    """Where a tool runs. Two arms, and the machine never learns which it got."""

    HOST = "host"
    CONTAINER = "container"


class TierUnavailable(RuntimeError):
    """`Tier.CONTAINER` was asked for and the image is not present.

    Raised rather than falling back, because a silent downgrade to the host is the failure this
    module exists to prevent: the caller believed it had isolation and executed model-written code
    beside its own files. An absent gate must not read as a passing one."""


def default_tier() -> Tier:
    """`HOST`, unless `EFFECTIVE_EXEC_TIER` says otherwise.

    An env var so a deployment can raise the floor for every caller at once without editing call
    sites, and an unrecognized value is an error rather than a silent host run — the same reason
    `TierUnavailable` exists."""
    match os.environ.get(TIER_ENV, "").strip().lower():
        case "" | "host":
            return Tier.HOST
        case "container":
            return Tier.CONTAINER
        case other:
            raise ValueError(f"{TIER_ENV}={other!r} is not a tier: use 'host' or 'container'")


@cache
def image_available(image: str = IMAGE) -> bool:
    """Is the pinned image built on this box? Cached — it cannot change mid-process usefully."""
    try:
        proc = subprocess.run(
            ["podman", "image", "exists", image], capture_output=True, timeout=30
        )
    except OSError, subprocess.SubprocessError:
        return False
    return proc.returncode == 0


class UnsafeTreePath(ValueError, Unretryable):
    """A tree key that leaves the tree's root, that no filesystem can write, or that some
    filesystem would write as the same file as another key. The key is refused alike on every
    attempt, so the task fails on the one that raised it; a key a model wrote reaches the model as
    a tool's refusal instead."""


def tree_path(name: str) -> str:
    """`name`, when it names one file inside a tree, in that file's one spelling.

    A tree's keys are written under a temporary root on the host and under `WORKDIR` in the
    container. An absolute key or one that climbs with `..` lands beside the files the predicate
    must not change. A second spelling of a file (`./x.py`, `x.py/`, `a//x.py`) is a different
    key to the static gate and to the tree, and the same file to the filesystem, so it could
    replace content the tree still shows. A name the filesystem cannot write would fail outside
    the tool boundary instead of being refused."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise UnsafeTreePath(f"tree key {name!r} is outside the tree")
    try:
        encoded = name.encode()
    except UnicodeError as unencodable:
        raise UnsafeTreePath(f"tree key {name!r} has no bytes to name a file by") from unencodable
    if len(encoded) > PATH_MAX:
        raise UnsafeTreePath(f"a tree key of {len(encoded)} bytes is longer than {PATH_MAX}")
    if (
        name == "."
        or path.as_posix() != name
        or "\0" in name
        or any(len(part.encode()) > NAME_MAX for part in path.parts)
    ):
        raise UnsafeTreePath(f"tree key {name!r} is not one file's canonical path in the tree")
    return name


def _folded(name: str) -> str:
    """`name` as a filesystem that folds case and normalizes Unicode compares it: the canonical
    caseless form."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def tree_paths(tree: Mapping[str, str]) -> None:
    """Every key of `tree` names one file inside it, and no two keys name one path: a file and its
    directory, or two names a case-folding or normalizing filesystem writes as one."""
    files: dict[str, str] = {}
    for name in tree:
        tree_path(name)
        if (first := files.setdefault(_folded(name), name)) != name:
            raise UnsafeTreePath(f"tree keys {first!a} and {name!a} can be written as one file")
    for name in tree:
        for parent in PurePosixPath(name).parents:
            if (file := files.get(_folded(parent.as_posix()))) is not None:
                raise UnsafeTreePath(
                    f"tree key {file!a} is both a file and the directory of {name!a}"
                )


NAME_MAX = 255
"""The longest path component, in bytes, the host and the container's filesystems accept."""

PATH_MAX = 1024
"""The longest key, in bytes: a quarter of the filesystems' 4096, which also holds each tier's
root, and the host's root is as long as `TMPDIR` makes it."""


def _archive(tree: Mapping[str, str]) -> bytes:
    """`tree` as an uncompressed tar of regular files, the one form both tiers unpack."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, content in sorted(tree.items()):
            data = content.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def _output(stdout: bytes | None, stderr: bytes | None) -> str:
    """What the command wrote, stdout then stderr, decoded the same way on either tier."""
    return ((stdout or b"") + (stderr or b"")).decode(errors="replace")


TIMED_OUT = 124
"""The exit code of a command stopped at its time limit, the one `timeout(1)` uses."""


def _timed_out(expired: subprocess.TimeoutExpired) -> tuple[int, str]:
    """A command past its limit, as a failed run: what it wrote, and that it was stopped."""
    return TIMED_OUT, _output(expired.output, expired.stderr) + "\ntimed out\n"


LIMITS: tuple[str, ...] = ("--memory", "2g", "--pids-limit", "256", "--cpus", "2")
"""What a container may consume, beside the time limit its caller sets.

The time limit bounds a command that runs too long; it does nothing about one that consumes the
box while it runs, and this executes code a model wrote. A fork bomb and a memory hog are the two
that reach the host through shared resources, so the pids and memory ceilings are the ones that
matter; the CPU share keeps a spinning command from taking the machine with it."""


_UNPACK = 'mkdir -p "$1" && cd "$1" && shift && tar -x && exec "$@"'
"""The container's entrypoint: unpack the tree from stdin into its first argument, then become the
command. The tree, the directory and the command arrive as stdin and argv, never as script text."""


def _run_container(
    tree: Mapping[str, str], command: Sequence[str], *, timeout: int, image: str
) -> tuple[int, str]:
    """Unpack and run, in a fresh egress-denied container.

    The command replaces the entrypoint's shell, so podman's exit code is the command's own and
    nothing the command prints can change it. `--init` makes a minimal init PID 1, so a signal the
    command sends itself ends it as on the host, as 128 plus the signal. Podman's own failures exit
    125, and 126 or 127 when the command cannot run; a command exiting with one of those reads the
    same, and none is a pass. At the time limit the container is removed, not left running."""
    if not image_available(image):
        raise TierUnavailable(
            f"{image!r} is not built, so container execution is unavailable. Build it with "
            f"`podman build -t {image} -f infra/code-agent/Dockerfile infra/code-agent`; this "
            f"refuses rather than running on the host, because a "
            f"caller asking for the container tier is asking not to execute model-written code "
            f"beside its own files."
        )
    entrypoint = ["sh", "-c", _UNPACK, "sh", WORKDIR]
    with tempfile.TemporaryDirectory() as directory:
        # Podman names the container here, so a limit that kills the client can remove it.
        cidfile = os.path.join(directory, "container-id")
        run = [
            "podman",
            "run",
            "--rm",
            "-i",
            "--init",
            "--network=none",
            *LIMITS,
            "--cidfile",
            cidfile,
        ]
        try:
            proc = subprocess.run(
                [*run, image, *entrypoint, *command],
                input=_archive(tree),
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as expired:
            remove = ["podman", "rm", "--force", "--cidfile", cidfile]
            subprocess.run(remove, capture_output=True, timeout=60)
            return _timed_out(expired)
    return proc.returncode, _output(proc.stdout, proc.stderr)


def _run_host(tree: Mapping[str, str], command: Sequence[str], *, timeout: int) -> tuple[int, str]:
    """Materialize into a temp directory and run there.

    The temp directory, never the real tree: the predicate must not be able to change the repo it
    is measuring. That is a correctness property and it is NOT isolation — the process still runs
    as this user, with this user's filesystem and network. The command gets no stdin, as in the
    container, and a signal that ends it reads as 128 plus the signal, as the container reports."""
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(fileobj=io.BytesIO(_archive(tree))) as archive:
            archive.extractall(directory, filter="data")
        try:
            proc = subprocess.run(
                [part.replace(PYTHON, sys.executable) for part in command],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                cwd=directory,
            )
        except subprocess.TimeoutExpired as expired:
            return _timed_out(expired)
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    return code, _output(proc.stdout, proc.stderr)


PYTHON = "python"
"""The interpreter placeholder a command uses, resolved per tier.

On the host it becomes `sys.executable`, so the suite runs under the interpreter driving the
machine rather than whatever `python` happens to be on `PATH` — the pin at `.python-version`
exists because that difference has cost this repo 15 tests. In the container it stays `python`,
which is the image's own pinned interpreter and the only one there."""


def run_tree_command(
    tree: Mapping[str, str],
    command: Sequence[str],
    *,
    timeout: int = 120,
    tier: Tier | None = None,
    image: str = IMAGE,
) -> tuple[int, str]:
    """Run `command` over a materialized `tree`, on the chosen tier. Returns `(exit_code, output)`.

    Both arms return the command's OWN exit code, so a caller cannot tell the tiers apart from the
    result — which is what makes the tier swappable rather than a fork in the caller's logic.

    The tree is checked by `tree_paths` before a tier is chosen, so neither arm writes outside
    its root or writes one file under two names."""
    tree_paths(tree)
    match tier if tier is not None else default_tier():
        case Tier.CONTAINER:
            return _run_container(tree, command, timeout=timeout, image=image)
        case Tier.HOST:
            return _run_host(tree, command, timeout=timeout)


__all__ = [
    "IMAGE",
    "PYTHON",
    "TIER_ENV",
    "Tier",
    "TierUnavailable",
    "UnsafeTreePath",
    "default_tier",
    "image_available",
    "run_tree_command",
    "tree_path",
    "tree_paths",
]
