"""Each witness sets its event while the other thread is still short of returning, so a test that
releases only after the event drives the call through its blocking path."""

import socket
import threading

import pytest
from _witness import WatchedCondition, WatchedLock, WatchedSocket
from test_race_shapes import HOLD, until


def _blocked(call) -> tuple[threading.Thread, list[object]]:
    returned: list[object] = []
    thread = threading.Thread(target=lambda: returned.append(call()))
    thread.start()
    return thread, returned


def _joined(thread: threading.Thread) -> None:
    thread.join(HOLD)
    assert not thread.is_alive(), "the witnessed call never returned"


def test_a_condition_witnesses_its_wait_before_a_notify():
    entered = threading.Event()
    ready = WatchedCondition(threading.Lock(), entered)
    woken: list[bool] = []

    def wait() -> bool:
        with ready:
            woken.append(ready.wait(HOLD))
        return True

    thread, _ = _blocked(wait)
    until(entered)
    with ready:
        ready.notify_all()
    _joined(thread)
    assert woken == [True]


def test_a_lock_witnesses_an_acquire_that_blocks_and_only_that():
    entered = threading.Event()
    lock = WatchedLock(threading.Lock(), entered)
    with lock:
        pass
    assert not entered.is_set()

    lock.acquire()
    thread, returned = _blocked(lambda: lock.acquire(True, HOLD))
    until(entered)
    lock.release()
    _joined(thread)
    assert returned == [True]
    assert lock.locked()


def test_a_lock_that_will_not_block_witnesses_nothing():
    entered = threading.Event()
    lock = WatchedLock(threading.Lock(), entered)
    lock.acquire()
    assert lock.acquire(False) is False
    assert not entered.is_set()


def test_a_condition_over_a_watched_lock_runs_its_wait():
    contended, waited = threading.Event(), threading.Event()
    ready = WatchedCondition(WatchedLock(threading.Lock(), contended), waited)

    def wait() -> bool:
        with ready:
            return ready.wait(HOLD)

    thread, returned = _blocked(wait)
    until(waited)
    with ready:
        ready.notify_all()
    _joined(thread)
    assert returned == [True]


@pytest.mark.parametrize("read", ["recv", "recv_into"])
def test_a_socket_witnesses_its_read_before_the_bytes_arrive(read):
    near, far = socket.socketpair()
    with near, far:
        entered = threading.Event()
        watched = WatchedSocket(near, entered)
        buffer = bytearray(4)
        reads = {"recv": lambda: watched.recv(4), "recv_into": lambda: watched.recv_into(buffer)}
        thread, returned = _blocked(reads[read])
        until(entered)
        far.sendall(b"ping")
        _joined(thread)
        assert {"recv": returned, "recv_into": [bytes(buffer)]}[read] == [b"ping"]
        assert watched.fileno() == near.fileno()
