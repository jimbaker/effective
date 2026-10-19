"""Witnesses: a wrapper whose blocking call sets an event as it enters, so a test orders itself
against another thread's wait by that event rather than by a pause.

| wraps                 | it witnesses                  | the event is set                       |
|-----------------------|-------------------------------|----------------------------------------|
| `threading.Condition` | `wait`                        | with the condition's lock still held   |
| `threading.Event`     | `wait`                        | before the wait                        |
| a lock                | an `acquire` that would block | after a non-blocking try finds it held |
| a socket              | `recv`, `recv_into`           | before the read                        |

Each event is set before the call blocks, and each wrapped resource keeps what a release sends in
that gap: a set event stays set, a notify needs the lock the waiter holds until its wait releases
it, a release lets the blocking acquire through, and bytes sent wait in the socket's buffer.
"""

import socket
import threading
from typing import Any


class WatchedCondition(threading.Condition):
    def __init__(self, lock: Any, entered: threading.Event) -> None:
        super().__init__(lock)
        self.entered = entered

    def wait(self, timeout: float | None = None) -> bool:
        self.entered.set()
        return super().wait(timeout)


class WatchedEvent(threading.Event):
    def __init__(self, entered: threading.Event) -> None:
        super().__init__()
        self.entered = entered

    def wait(self, timeout: float | None = None) -> bool:
        self.entered.set()
        return super().wait(timeout)


class WatchedLock:
    """A lock, delegated to, since `threading.Lock` cannot be subclassed."""

    def __init__(self, lock: Any, entered: threading.Event) -> None:
        self._lock = lock
        self.entered = entered

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(False):
            return True
        if not blocking:
            return False
        self.entered.set()
        return self._lock.acquire(True, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *raised: object) -> None:
        self.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lock, name)


class WatchedSocket:
    def __init__(self, sock: socket.socket, entered: threading.Event) -> None:
        self._sock = sock
        self.entered = entered

    def recv(self, *args: Any) -> bytes:
        self.entered.set()
        return self._sock.recv(*args)

    def recv_into(self, *args: Any) -> int:
        self.entered.set()
        return self._sock.recv_into(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)
