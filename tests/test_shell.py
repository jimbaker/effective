"""A shell command a cancel stops: the group it started, the output so far, and the grace."""

import os
import select
import threading
from pathlib import Path

from effective.cancel import Cancelled, CancelToken
from effective.interpreters.shell import TIMED_OUT, Shelled, run_shell

SAFETY_NET = 10
OUTLASTING = 3 * SAFETY_NET  # a command's seconds; a missed kill fails a check before it ends


def test_a_command_that_exits_answers_its_code_and_interleaved_output():
    assert run_shell("echo one; echo two >&2; echo three; exit 3") == Shelled(
        3, "one\ntwo\nthree\n"
    )


def test_a_command_past_its_timeout_answers_the_timeout_code():
    shelled = run_shell(f"echo started; exec sleep {OUTLASTING}", timeout_s=0.2)
    assert shelled == Shelled(TIMED_OUT, "started\n")


def test_a_cancelled_token_runs_nothing(tmp_path: Path):
    token = CancelToken()
    token.cancel()
    marker = tmp_path / "ran"
    assert run_shell(f"touch {marker}", cancel=token) == Cancelled()
    assert not marker.exists()


def readable(fd: int) -> bool:
    """Whether `fd` reads ready within the safety net."""
    ready, _, _ = select.select([fd], [], [], SAFETY_NET)
    return bool(ready)


def started(marker: Path) -> int:
    """A FIFO at `marker`, opened to read, which a command writes once it has started."""
    os.mkfifo(marker)
    return os.open(marker, os.O_RDONLY | os.O_NONBLOCK)


class Canceller(threading.Thread):
    """Cancels `token` once the command has written the FIFO `marker`, then writes a line to the
    FIFO `release` when there is one, so a command reading it runs on only after the stop's
    SIGTERM was sent. `wrote` says whether the command wrote `marker`."""

    def __init__(self, marker: Path, token: CancelToken, release: Path | None = None) -> None:
        super().__init__(name="face", daemon=True)
        self.fifo, self.token, self.release = started(marker), token, release
        self.wrote = False
        self.start()

    def run(self) -> None:
        self.wrote = readable(self.fifo)
        os.close(self.fifo)
        self.token.cancel()
        if self.release is not None:
            with self.release.open("w") as fifo:
                fifo.write("go\n")


def exits(pid: int) -> bool:
    """Whether the process `pid` has exited, or exits before the safety net."""
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        return readable(pidfd)
    finally:
        os.close(pidfd)


def test_a_cancel_stops_the_whole_group_and_keeps_the_output_so_far(tmp_path: Path):
    marker, pidfile, finished = tmp_path / "started", tmp_path / "background", tmp_path / "ended"
    token = CancelToken()
    watcher = Canceller(marker, token)
    result = run_shell(
        f"echo partial; sleep {OUTLASTING} >/dev/null 2>&1 & echo $! > {pidfile}; "
        f"echo > {marker}; wait; touch {finished}",
        cancel=token,
    )
    watcher.join(SAFETY_NET)
    assert watcher.wrote, "the command never wrote its marker"
    assert result == Cancelled(partial="partial\n")
    assert not finished.exists(), "the cancel waited for the command"
    assert exits(int(pidfile.read_text())), "the command's background child outlived the cancel"


def test_a_command_surviving_sigterm_gets_sigkill_after_its_grace_period(tmp_path: Path):
    marker, termed, finished = tmp_path / "started", tmp_path / "termed", tmp_path / "ended"
    token = CancelToken()
    watcher = Canceller(marker, token)
    result = run_shell(
        f"trap 'touch {termed}' TERM; echo > {marker}; "
        f"sleep {OUTLASTING} >/dev/null 2>&1 & wait; sleep {OUTLASTING} >/dev/null 2>&1; "
        f"touch {finished}",
        cancel=token,
        grace_s=0.2,
    )
    watcher.join(SAFETY_NET)
    assert watcher.wrote, "the command never wrote its marker"
    assert isinstance(result, Cancelled)
    assert termed.exists(), "SIGKILL arrived with the SIGTERM"
    assert not finished.exists(), "SIGKILL never followed the grace period"


def test_a_command_that_exits_after_a_cancel_reports_its_own_exit(tmp_path: Path):
    marker, release = tmp_path / "started", tmp_path / "release"
    os.mkfifo(release)
    token = CancelToken()
    watcher = Canceller(marker, token, release)
    result = run_shell(
        f"trap '' TERM; echo > {marker}; read go < {release}; echo finished; exit 0",
        cancel=token,
        grace_s=10,
    )
    watcher.join(SAFETY_NET)
    assert watcher.wrote, "the command never wrote its marker"
    assert token.cancelled
    assert result == Shelled(0, "finished\n")


def test_a_descendant_that_ignores_sigterm_and_lets_go_of_the_pipe_is_killed(tmp_path: Path):
    marker, pidfile = tmp_path / "started", tmp_path / "descendant"
    token = CancelToken()
    watcher = Canceller(marker, token)
    result = run_shell(
        f"(trap '' TERM; exec sleep {OUTLASTING}) >/dev/null 2>&1 & echo $! > {pidfile}; "
        f"echo > {marker}; sleep {OUTLASTING} >/dev/null 2>&1",
        cancel=token,
        grace_s=0.3,
    )
    watcher.join(SAFETY_NET)
    assert watcher.wrote, "the command never wrote its marker"
    assert isinstance(result, Cancelled)
    assert exits(int(pidfile.read_text())), "the kill after the grace never reached the descendant"


def test_a_command_that_dies_of_its_own_signal_reports_it_after_a_cancel(tmp_path: Path):
    """The command ignores the stop's SIGTERM and blocks on a FIFO the test opens only once
    `cancel()` has returned, so its SIGSEGV always lands after the cancel was delivered."""
    marker, release = tmp_path / "started", tmp_path / "release"
    os.mkfifo(release)
    token = CancelToken()
    watcher = Canceller(marker, token, release)
    result = run_shell(
        f"trap '' TERM; echo > {marker}; read go < {release}; kill -SEGV $$",
        cancel=token,
        grace_s=10,
    )
    watcher.join(SAFETY_NET)
    assert watcher.wrote, "the command never wrote its marker"
    assert token.cancelled
    assert result == Shelled(-11, "")
