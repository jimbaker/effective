"""The Node facade: one persistent, sandboxed elkjs process speaking JSON lines.

**node runs in a container, always**: the standing supply-chain posture, not a judgment about
elkjs. `just elk-image` builds `infra/elkjs/Containerfile`, which does the
`npm ci --ignore-scripts` and the two bundle-hash checks at BUILD time, so a runnable image implies
a verified artifact. At run time the worker gets `--network=none`, a read-only rootfs, no host
mount, and no capabilities: a graph goes in on stdin, geometry comes out on stdout. The host
fallback (`EFFECTIVE_ELK_HOST=1`) exists for a device without Podman and runs the npm graph
unsandboxed — it is the exception, spelled loudly, exactly like `just formal-verify-host`.

This facade is for tests, snapshots, static export, and server-side rendering. The interactive
dashboard's default path is the *same* elkjs bundle in a browser Web Worker, consuming the *same*
`envelope()` payload: one layout contract, two execution paths. The pinned
`elk-worker.min.js` hash in `infra/elkjs/PIN.txt` is what keeps them on one engine build.

Everything here is failure-handling around one line of protocol: a request that times out or
answers out of order kills the process and restarts it on the next call, because a layout engine
holding half a response is worth less than a cold start.
"""

import json
import os
import selectors
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Any, Self

from effective.graphlayout.elk import from_elk, to_elk
from effective.graphlayout.model import Geometry, LayoutGraph

IMAGE = "effective-elk:0.12.0"
"""Tagged `<package>:<version>`, matching `infra/elkjs/PIN.txt`. A bump changes the tag, so a
stale image cannot silently serve a new pin."""

INFRA = Path(__file__).resolve().parents[3] / "infra" / "elkjs"
WORKER = INFRA / "layout_worker.mjs"

CONTAINER_ARGS = (
    "run",
    "--rm",
    "-i",
    "--pull=never",  # never reach the network at run time; the image is built, not fetched
    "--network=none",
    "--read-only",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
)


class ElkLayoutError(RuntimeError):
    """The layout engine failed, timed out, or broke protocol."""


def _host_mode() -> bool:
    return os.environ.get("EFFECTIVE_ELK_HOST", "") not in ("", "0")


def available() -> bool:
    """Whether this machine can lay out a graph — the test skip-gate.

    Deliberately not a "can I find node" check: in the default (container) mode the answer is
    whether the *pinned image* exists, so a machine with node installed but no built image skips
    rather than silently falling back to an unsandboxed run."""
    if _host_mode():
        return (
            shutil.which("node") is not None
            and WORKER.exists()
            and (INFRA / "node_modules" / "elkjs").exists()
        )
    podman = shutil.which("podman")
    if podman is None:
        return False
    probe = subprocess.run([podman, "image", "exists", IMAGE], capture_output=True, check=False)
    return probe.returncode == 0


class ElkJs:
    """A persistent elkjs worker. Use as a context manager, or rely on `close()`.

    One process serves many layouts because process start-up (a container spawn) dominates a
    small graph's actual layout time by an order of magnitude. Access is serialized by a lock:
    the protocol is one request per line and one response per line, so concurrency here would
    interleave two conversations on one pipe."""

    def __init__(self, *, timeout: float = 10.0, image: str = IMAGE) -> None:
        self.timeout = timeout
        self.image = image
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr: Any = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._next_id = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _command(self) -> list[str]:
        if _host_mode():
            return ["node", str(WORKER)]
        return ["podman", *CONTAINER_ARGS, self.image]

    def _start(self) -> subprocess.Popen[bytes]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        # A file rather than a PIPE, deliberately: a pipe nobody drains deadlocks a chatty
        # worker, and this one's lifetime is the process's, not a block's.
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115
        self._buffer = bytearray()
        self._process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
        )
        return self._process

    def _diagnostics(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", "replace").strip()
        except OSError, ValueError:  # already closed
            return ""

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout) if process else ():
            if stream is not None:
                stream.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def _read_line(self, process: subprocess.Popen[bytes], deadline: float) -> bytes:
        """One response line, or a timeout.

        Blocking `readline()` on a pipe has no deadline, and a worker that wedges would hang the
        caller forever; `selectors` gives the wait a bound without a reader thread whose lifetime
        would then outlive the request it belongs to."""
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - monotonic()
                if remaining <= 0 or not selector.select(timeout=remaining):
                    raise TimeoutError("elkjs worker did not answer in time")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise ElkLayoutError(f"elkjs worker exited: {self._diagnostics()}")
                self._buffer.extend(chunk)
        # Keep the remainder: a read can return more than one line, and a dropped tail would
        # desync every later response by one.
        line, _, rest = bytes(self._buffer).partition(b"\n")
        self._buffer = bytearray(rest)
        return line

    def request(self, elk_graph: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
        """One layout round trip: the raw ELK result and the engine version that produced it."""
        with self._lock:
            process = self._start()
            self._next_id += 1
            request_id = str(self._next_id)
            payload = json.dumps({"id": request_id, "graph": elk_graph}) + "\n"
            try:
                assert process.stdin is not None
                process.stdin.write(payload.encode("utf-8"))
                process.stdin.flush()
                line = self._read_line(process, monotonic() + self.timeout)
            except (TimeoutError, BrokenPipeError, OSError) as exc:
                self.close()  # a worker mid-response is worth less than a cold start
                raise ElkLayoutError(f"elkjs layout failed: {exc}") from exc

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise ElkLayoutError(f"elkjs worker broke protocol: {line!r}") from exc

            if (answered := response.get("id")) != request_id:
                self.close()
                raise ElkLayoutError(
                    f"elkjs response out of order: wanted {request_id!r}, got {answered!r}"
                )
            if not response.get("ok"):
                raise ElkLayoutError(f"elkjs layout failed: {response.get('error')}")
            return response["result"], response.get("engine_version")

    def layout(self, graph: LayoutGraph, *, options: Mapping[str, str] | None = None) -> Geometry:
        """`LayoutGraph` → `Geometry`. The whole engine seam, in one method."""
        result, version = self.request(to_elk(graph, options=options))
        return from_elk(result, engine="elkjs", engine_version=version)


_ENGINE: ElkJs | None = None
_ENGINE_LOCK = threading.Lock()


def engine() -> ElkJs:
    """The process-wide worker — started on first use, reused after.

    A module singleton because the expensive thing is the process, not the object, and every
    caller in one server process wants the same one."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = ElkJs()
        return _ENGINE


def layout(graph: LayoutGraph, *, options: Mapping[str, str] | None = None) -> Geometry:
    """Lay out a graph on the shared worker."""
    return engine().layout(graph, options=options)


def shutdown() -> None:
    """Stop the shared worker (tests, and a server's lifespan teardown)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            _ENGINE.close()
            _ENGINE = None


__all__ = [
    "CONTAINER_ARGS",
    "IMAGE",
    "ElkJs",
    "ElkLayoutError",
    "available",
    "engine",
    "layout",
    "shutdown",
]
