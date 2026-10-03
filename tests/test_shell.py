"""A shell command a cancel stops: the group it started, the output so far, and the grace."""

import os
import threading
import time
from pathlib import Path

from effective.cancel import Cancelled, CancelToken
from effective.interpreters.shell import TIMED_OUT, Shelled, run_shell


def test_a_command_that_exits_answers_its_code_and_interleaved_output():
    assert run_shell("echo one; echo two >&2; echo three; exit 3") == Shelled(
        3, "one\ntwo\nthree\n"
    )


def test_a_command_past_its_timeout_answers_the_timeout_code():
    shelled = run_shell("echo started; exec sleep 30", timeout_s=0.2)
    assert shelled == Shelled(TIMED_OUT, "started\n")


def test_a_cancelled_token_runs_nothing(tmp_path: Path):
    token = CancelToken()
    token.cancel()
    marker = tmp_path / "ran"
    assert run_shell(f"touch {marker}", cancel=token) == Cancelled()
    assert not marker.exists()


def cancel_once(marker: Path, token: CancelToken) -> threading.Thread:
    """Cancel `token` from another thread once the command has written `marker`."""

    def watch() -> None:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        token.cancel()

    watcher = threading.Thread(target=watch, name="face")
    watcher.start()
    return watcher


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_cancel_stops_the_whole_group_and_keeps_the_output_so_far(tmp_path: Path):
    marker, pidfile = tmp_path / "started", tmp_path / "background"
    token = CancelToken()
    watcher = cancel_once(marker, token)
    began = time.monotonic()
    result = run_shell(
        f"echo partial; sleep 30 & echo $! > {pidfile}; touch {marker}; wait", cancel=token
    )
    watcher.join()
    assert result == Cancelled(partial="partial\n")
    assert time.monotonic() - began < 5, "the cancel waited for the command"
    background = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while alive(background) and time.monotonic() < deadline:  # init reaps it once it dies
        threading.Event().wait(0.01)
    assert not alive(background), "the command's background child outlived the cancel"


def test_a_command_ignoring_sigterm_is_killed_after_the_grace(tmp_path: Path):
    marker = tmp_path / "started"
    token = CancelToken()
    watcher = cancel_once(marker, token)
    began = time.monotonic()
    result = run_shell(f"trap '' TERM; touch {marker}; sleep 30", cancel=token, grace_s=0.2)
    watcher.join()
    assert isinstance(result, Cancelled)
    assert time.monotonic() - began < 5, "SIGKILL never followed the grace"


def test_a_command_that_exits_after_a_cancel_reports_its_own_exit(tmp_path: Path):
    marker = tmp_path / "started"
    token = CancelToken()
    watcher = cancel_once(marker, token)
    result = run_shell(
        f"trap '' TERM; touch {marker}; sleep 0.3; echo finished; exit 0", cancel=token, grace_s=5
    )
    watcher.join()
    assert result == Shelled(0, "finished\n")


def test_a_descendant_that_ignores_sigterm_and_lets_go_of_the_pipe_is_killed(tmp_path: Path):
    marker, pidfile = tmp_path / "started", tmp_path / "descendant"
    token = CancelToken()
    watcher = cancel_once(marker, token)
    result = run_shell(
        f"(trap '' TERM; exec sleep 30) >/dev/null 2>&1 & echo $! > {pidfile}; "
        f"touch {marker}; sleep 30",
        cancel=token,
        grace_s=0.3,
    )
    watcher.join()
    assert isinstance(result, Cancelled)
    descendant = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while alive(descendant) and time.monotonic() < deadline:
        threading.Event().wait(0.05)
    assert not alive(descendant), "the kill after the grace never reached the descendant"


def test_a_command_that_dies_of_its_own_signal_reports_it_after_a_cancel(tmp_path: Path):
    """The command ignores the stop's SIGTERM and blocks on a FIFO the test opens only once
    `cancel()` has returned, so its SIGSEGV always lands after the cancel was delivered."""
    marker, release = tmp_path / "started", tmp_path / "release"
    os.mkfifo(release)
    token = CancelToken()

    def cancel_then_release() -> None:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        token.cancel()  # runs the stop, so the SIGTERM has been sent when this returns
        with release.open("w") as fifo:
            fifo.write("go\n")

    watcher = threading.Thread(target=cancel_then_release, name="face")
    watcher.start()
    result = run_shell(
        f"trap '' TERM; touch {marker}; read go < {release}; kill -SEGV $$",
        cancel=token,
        grace_s=10,
    )
    watcher.join()
    assert token.cancelled
    assert result == Shelled(-11, "")
