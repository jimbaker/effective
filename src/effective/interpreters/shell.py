"""A shell command a `CancelToken` can stop while it runs.

The command runs in a session of its own, so a cancel reaches everything it started: the process
group gets SIGTERM, then SIGKILL after `grace_s` if anything is still running. stderr joins stdout
in one pipe, which keeps the interleaving a terminal would show.

| the command             | the result                                  |
|-------------------------|---------------------------------------------|
| exits                   | `Shelled(exit_code, output)`                |
| outlives `timeout_s`    | `Shelled(124, output)`, the group killed    |
| is killed by a cancel   | `Cancelled(partial=output)`                 |
| exits after a cancel    | `Shelled(exit_code, output)`, its own exit  |
"""

import os
import signal
import subprocess
import threading
from contextlib import nullcontext, suppress
from dataclasses import dataclass

from effective.cancel import Cancelled, CancelToken

TIMED_OUT = 124
"""The exit code `timeout(1)` reports, which a model reading the output already knows."""

STOPPING = frozenset({signal.SIGTERM, signal.SIGKILL})
"""The signals a stop sends; a command that died of another died of its own accord."""


@dataclass(frozen=True, slots=True)
class Shelled:
    exit_code: int
    output: str


def _signal_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    with suppress(ProcessLookupError):  # the group has already exited
        os.killpg(process.pid, sig)


def run_shell(
    command: str,
    *,
    cancel: CancelToken | None = None,
    grace_s: float = 2.0,
    timeout_s: float | None = None,
) -> Shelled | Cancelled:
    if cancel is not None and cancel.cancelled:
        return Cancelled()
    process = subprocess.Popen(
        ["/bin/sh", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    stopped = threading.Event()
    kill = threading.Timer(grace_s, _signal_group, (process, signal.SIGKILL))
    kill.daemon = True

    once = threading.Lock()

    def stop() -> None:
        with once:  # a token reset and cancelled again stops this command once
            if stopped.is_set():
                return
            stopped.set()
        _signal_group(process, signal.SIGTERM)
        kill.start()

    try:
        with cancel.on_cancel(stop) if cancel is not None else nullcontext():
            try:
                output, _ = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                output, _ = process.communicate()
                return Shelled(TIMED_OUT, output)
    finally:
        if not stopped.is_set():  # after a stop, the kill still reaches a lingering descendant
            kill.cancel()
    if stopped.is_set() and -process.returncode in STOPPING:
        return Cancelled(partial=output)
    return Shelled(process.returncode, output)
