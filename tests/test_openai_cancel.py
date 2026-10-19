"""A Responses call given a `CancelToken`: streamed, and closed by a cancel from another thread."""

import asyncio
import concurrent.futures
import gc
import inspect
import json
import os
import socket
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from _witness import WatchedSocket

from effective.cancel import Cancelled, CancelToken
from effective.domain import AskLLM
from effective.interpreters.aio import LoopThread, Stopped, shared_loop
from effective.interpreters.openai import (
    GPT56_LUNA,
    AsyncResponsesTurnCaller,
    ResponsesTurnCaller,
    _WireTurn,
)
from effective.react import AssistantTurn

ASK = AskLLM(messages=[{"role": "user", "content": "go"}], response_schema=AssistantTurn)


class _Stream:
    """Sends `deltas`, then blocks the reader until `release`, as a socket read does: closing a
    socket does not wake a read blocked on it in another thread."""

    def __init__(self, deltas: list[str], hang: bool) -> None:
        self.deltas = deltas
        self.hang = hang
        self.reading = threading.Event()
        self.closed = threading.Event()
        self.release = threading.Event()
        self.unblocked = threading.Event()

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed.set()

    def __iter__(self) -> Any:
        for delta in self.deltas:
            yield SimpleNamespace(type="response.output_text.delta", delta=delta)
        if self.hang:
            self.reading.set()
            self.release.wait(10)
            self.unblocked.set()
            raise ConnectionError("the connection was reset")

    def close(self) -> None:
        self.closed.set()

    def get_final_response(self) -> Any:
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5, input_tokens_details=SimpleNamespace(cached_tokens=0)
        )
        answered = _WireTurn(thought="t", tool=None, answer="".join(self.deltas))
        return SimpleNamespace(output=[], output_parsed=answered, usage=usage)


class _Client:
    def __init__(self, stream: _Stream) -> None:
        self.opened = 0
        self._stream = stream
        self.responses = SimpleNamespace(stream=self._open)

    def _open(self, **_: Any) -> _Stream:
        self.opened += 1
        return self._stream


def caller(client: Any, token: CancelToken) -> ResponsesTurnCaller:
    return ResponsesTurnCaller(
        client=client, system_prompt="s", model="gpt-5.6-luna", price=GPT56_LUNA, cancel=token
    )


def test_a_finished_stream_answers_the_turn():
    stream = _Stream(["hel", "lo"], hang=False)
    turn, usage = caller(_Client(stream), CancelToken())(ASK)
    assert isinstance(turn, AssistantTurn)
    assert turn.answer == "hello"
    assert usage.prompt_tokens == 10


def test_a_cancel_closes_the_stream_and_answers_what_arrived():
    stream, token = _Stream(["hel"], hang=True), CancelToken()

    def face() -> None:
        stream.reading.wait(10)
        token.cancel()

    pressed = threading.Thread(target=face, name="face")
    pressed.start()
    turn, usage = caller(_Client(stream), token)(ASK)
    held = not stream.unblocked.is_set()
    pressed.join()
    stream.release.set()
    assert turn == Cancelled(partial="hel")
    assert held, "the op waited for a read the close cannot wake"
    assert stream.closed.is_set()
    assert usage.prompt_tokens == 0


def test_a_call_after_the_cancel_sends_nothing():
    client, token = _Client(_Stream(["hel"], hang=False)), CancelToken()
    token.cancel()
    turn, _ = caller(client, token)(ASK)
    assert turn == Cancelled(partial="")
    assert client.opened == 0


def test_a_cancel_then_a_reset_still_answers_cancelled():
    """The op branches on its own stop, which a reset of the token cannot undo."""
    stream, token = _Stream(["hel"], hang=True), CancelToken()

    def face() -> None:
        stream.reading.wait(10)
        token.cancel()
        token.reset()

    pressed = threading.Thread(target=face, name="face")
    pressed.start()
    turn, _ = caller(_Client(stream), token)(ASK)
    pressed.join()
    stream.release.set()
    assert turn == Cancelled(partial="hel")


# --- the real SDK over a real socket ---------------------------------------------------------

RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "m",
    "output": [],
    "status": "in_progress",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
ITEM = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "in_progress",
    "content": [],
}
ANSWER = json.dumps({"thought": "t", "tool": None, "answer": "hello"})


def _event(kind: str, **fields: Any) -> bytes:
    data = json.dumps({"type": kind, "sequence_number": 0, **fields})
    return b"event: " + kind.encode() + b"\ndata: " + data.encode() + b"\n\n"


def _delta(text: str) -> bytes:
    return _event(
        "response.output_text.delta",
        output_index=0,
        content_index=0,
        item_id="msg_1",
        delta=text,
        logprobs=[],
    )


OPENING = [
    _event("response.created", response=RESPONSE),
    _event("response.output_item.added", output_index=0, item=ITEM),
    _event(
        "response.content_part.added",
        output_index=0,
        content_index=0,
        item_id="msg_1",
        part={"type": "output_text", "text": "", "annotations": []},
    ),
]
USAGE = {
    "input_tokens": 3,
    "output_tokens": 2,
    "total_tokens": 5,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}
COMPLETED = _event(
    "response.completed",
    response={
        **RESPONSE,
        "status": "completed",
        "output": [
            {
                **ITEM,
                "status": "completed",
                "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
            }
        ],
        "usage": USAGE,
    },
)


INCOMPLETE = _event(
    "response.incomplete",
    response={
        **RESPONSE,
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
)


class _Provider:
    """A local `/v1/responses` stream. The first call stalls until `resume` at the point `first`
    names; every later call streams the whole answer.

    | `first`    | the first call stalls                                       |
    |------------|-------------------------------------------------------------|
    | `silent`   | after one delta, as a model reasoning does                  |
    | `headers`  | before its response headers, as a queued request does       |
    | `end`      | after `response.completed`, before the stream's last chunk  |
    | `ended`    | after `response.incomplete`, before the stream's last chunk |
    | `capped`   | never: it sends `response.incomplete` and ends the stream   |
    """

    def __init__(
        self, first: Literal["silent", "headers", "end", "ended", "capped"] = "silent"
    ) -> None:
        self.calls = 0
        self.silent = threading.Event()
        self.resume = threading.Event()
        self.client_gone = threading.Event()
        self.replied = threading.Event()
        self.left = threading.Event()
        self.released = threading.Event()
        self.spoke = threading.Event()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                provider.calls += 1
                stall = first if provider.calls == 1 else None
                if stall == "headers":
                    provider.silent.set()
                    provider.resume.wait(SAFETY_NET)
                    provider.released.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(chunk: bytes) -> None:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()

                try:
                    provider.reply(send, stall)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    provider.client_gone.set()
                finally:
                    if stall is not None:
                        provider.replied.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def heard(self) -> bool:
        """Whether the first call's reply ended on a client that had left, read once it ended."""
        assert self.replied.wait(SAFETY_NET), "the first reply never ended"
        return self.client_gone.is_set()

    def reply(self, send: Callable[[bytes], None], stall: str | None) -> None:
        """Everything but the last chunk; a `silent` stall never completes."""
        for event in OPENING:
            send(event)
        send(_delta(ANSWER[:5]))
        if stall == "capped":
            send(INCOMPLETE)
            return
        if stall == "ended":  # the output cap reached, which is no answer
            send(INCOMPLETE)
            self.silent.set()
            self.resume.wait(SAFETY_NET)
            return
        if stall == "headers":  # the rest of the reply, once the client's readers have ended
            self.left.wait(SAFETY_NET)
            for _ in range(40):
                send(_delta("x"))
        if stall == "silent":
            self.silent.set()
            self.resume.wait(SAFETY_NET)
            self.spoke.set()
            for _ in range(20):
                send(_delta("x"))
            return
        send(_delta(ANSWER[5:]))
        send(COMPLETED)
        if stall == "end":
            self.silent.set()
            self.resume.wait(SAFETY_NET)


type Readers = tuple[set[threading.Thread], set[asyncio.Task[Any]]]

SAFETY_NET = 60
"""Seconds a wait in these pins allows for an event a passing run delivers at once: a join, a
watched delta, a silent provider's resume. Only a run whose event never comes reaches it."""


def _readers() -> Readers:
    """The threaded caller's reader threads, and the tasks on the asyncio callers' loop."""

    async def tasks() -> set[asyncio.Task[Any]]:
        return asyncio.all_tasks() - {asyncio.current_task()}

    threads = {thread for thread in threading.enumerate() if thread.name == "responses-stream"}
    return threads, shared_loop().run(tasks())


def _join(before: Readers) -> Readers:
    """Join every reader started since `before`, and answer the ones still running."""
    threads, tasks = _readers()
    for thread in threads - before[0]:
        thread.join(SAFETY_NET)

    async def joined(started: set[asyncio.Task[Any]]) -> set[asyncio.Task[Any]]:
        if started:
            await asyncio.wait(started, timeout=SAFETY_NET)
        return {task for task in started if not task.done()}

    running = {thread for thread in threads - before[0] if thread.is_alive()}
    return running, shared_loop().run(joined(tasks - before[1]))


type Flavor = Literal["sync", "async"]
FLAVORS: list[Flavor] = ["sync", "async"]
"""Each real-socket pin runs against the threaded caller and the asyncio one."""


class _ReadingAfter(threading.Event):
    """An event a client socket's read sets only once `after` is set, so a wait on it ends in a
    read begun after `after`."""

    def __init__(self, after: threading.Event | None) -> None:
        super().__init__()
        self.after = after

    def set(self) -> None:
        if self.after is None or self.after.is_set():
            super().set()


def _reading(monkeypatch: pytest.MonkeyPatch, after: threading.Event | None = None) -> Any:
    """An event the threaded caller's socket sets as it enters a read begun once `after` is set.
    The asyncio caller needs none: its cancel is a callback on the loop its reader runs on, so it
    lands where the reader next awaits."""
    reading = _ReadingAfter(after)
    connect = socket.create_connection
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: WatchedSocket(connect(*a, **k), reading)
    )
    return reading


def _client(flavor: Flavor, provider: _Provider) -> Any:
    openai = pytest.importorskip("openai")
    kind = openai.OpenAI if flavor == "sync" else openai.AsyncOpenAI
    # a worker stalled under load connects late, and no pin here is about the connect
    timeout = openai.Timeout(600, connect=60)
    return kind(base_url=provider.url, api_key="unused", max_retries=0, timeout=timeout)


def _caller(flavor: Flavor, client: Any, token: CancelToken) -> ResponsesTurnCaller:
    kind = ResponsesTurnCaller if flavor == "sync" else AsyncResponsesTurnCaller
    return kind(
        client=client, system_prompt="s", model="gpt-5.6-luna", price=GPT56_LUNA, cancel=token
    )


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_cancel_on_a_pooled_connection_ends_the_reader_and_tells_the_provider(
    flavor: Flavor, monkeypatch: pytest.MonkeyPatch
):
    """Measured by a review: with the reader blocked in `recv` when the cancel closed the stream,
    and the next call on the same client, the reader stayed asleep and the provider wrote its
    whole reply. The face presses Esc once the reader has taken the first delta and, on the
    threaded caller, entered its next read; the asyncio caller's cancel lands at its next await."""
    provider = _Provider()
    pooled = _client(flavor, provider)
    taken, token = threading.Event(), CancelToken()
    wrap = _Watched if flavor == "sync" else _AsyncWatched
    in_read = _reading(monkeypatch, after=taken) if flavor == "sync" else taken
    seen: list[bool] = []

    def stream(**kwargs: Any) -> _Watched:
        return wrap(pooled.responses.stream(**kwargs), taken, DELTA)

    client = SimpleNamespace(responses=SimpleNamespace(stream=stream))
    before = _readers()

    def face() -> None:
        seen.append(in_read.wait(SAFETY_NET))
        token.cancel()

    pressed = threading.Thread(target=face, name="face")
    pressed.start()
    try:
        turn, _ = _caller(flavor, client, token)(ASK)
        pressed.join()
        assert seen == [True], "the face pressed Esc before the reader was in its next read"
        assert turn == Cancelled(partial=ANSWER[:5])
        following, _ = _caller(flavor, client, CancelToken())(ASK)
        assert isinstance(following, AssistantTurn)
        assert following.answer == "hello"
        assert _join(before) == (set(), set()), "the reader outlived the cancel"
        assert not provider.spoke.is_set(), "the reader ended only once the provider spoke"
    finally:
        provider.resume.set()
    assert provider.heard(), "the provider never heard the client leave"
    provider.server.shutdown()


class _Watched:
    """The SDK's stream manager, with a signal once the reader has taken an event of one of
    `types`: set when the reader asks for the event after it, so the reader has finished with that
    one."""

    def __init__(self, manager: Any, completed: threading.Event, types: frozenset[str]) -> None:
        self.manager = manager
        self.completed = completed
        self.types = types

    def __enter__(self) -> Any:
        return _WatchedStream(self.manager.__enter__(), self.completed, self.types)

    def __exit__(self, *raised: object) -> Any:
        return self.manager.__exit__(*raised)


class _WatchedStream:
    def __init__(self, stream: Any, completed: threading.Event, types: frozenset[str]) -> None:
        self._stream = stream
        self._completed = completed
        self._types = types

    def __getattr__(self, name: str) -> Any:  # `_sever` reaches `_response` through here
        return getattr(self._stream, name)

    def __iter__(self) -> Any:
        for event in self._stream:
            yield event
            if event.type in self._types:
                self._completed.set()


class _AsyncWatched(_Watched):
    async def __aenter__(self) -> Any:
        return _AsyncWatchedStream(await self.manager.__aenter__(), self.completed, self.types)

    async def __aexit__(self, *raised: object) -> Any:
        return await self.manager.__aexit__(*raised)


class _AsyncWatchedStream(_WatchedStream):
    async def __aiter__(self) -> Any:
        async for event in self._stream:
            yield event
            if event.type in self._types:
                self._completed.set()


WATCHED = frozenset({"response.completed", "response.incomplete"})
DELTA = frozenset({"response.output_text.delta"})


def _cancelled_at_the_stall(
    flavor: Flavor, first: Literal["headers", "end", "ended"], monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, float, bool]:
    """One call cancelled during the provider's stall: the turn, its usage, whether the op
    returned while the provider still held the headers, and whether the provider heard the client
    leave.

    | `first`   | the face presses Esc                                            |
    |-----------|-----------------------------------------------------------------|
    | `headers` | once the stall holds the headers, and the threaded reader reads |
    | `end`     | once the reader has taken `response.completed`                  |
    | `ended`   | once the reader has taken `response.incomplete`                 |
    """
    provider = _Provider(first)
    client = _client(flavor, provider)
    completed, token = threading.Event(), CancelToken()
    opened = client.responses.stream
    wrap = _Watched if flavor == "sync" else _AsyncWatched
    watched = SimpleNamespace(stream=lambda **kwargs: wrap(opened(**kwargs), completed, WATCHED))
    before = _readers()
    in_read = _reading(monkeypatch) if flavor == "sync" else provider.silent

    def face() -> None:
        if first in ("end", "ended"):
            completed.wait(SAFETY_NET)
        else:
            provider.silent.wait(SAFETY_NET)
            in_read.wait(SAFETY_NET)
        token.cancel()

    pressed = threading.Thread(target=face, name="face")
    pressed.start()
    try:
        turn, usage = _caller(flavor, SimpleNamespace(responses=watched), token)(ASK)
        held = not provider.released.is_set()
        pressed.join()
    finally:
        provider.resume.set()
        running = _join(before)
        provider.left.set()
        heard = provider.heard()
        provider.server.shutdown()
    assert running == (set(), set()), "the reader outlived the call"
    return turn, usage, held, heard


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_cancel_before_the_first_byte_returns_at_once(
    flavor: Flavor, monkeypatch: pytest.MonkeyPatch
):
    turn, _, held, heard = _cancelled_at_the_stall(flavor, "headers", monkeypatch)
    assert turn == Cancelled(partial="")
    assert held, "the op waited for headers the stop could not reach"
    assert heard, "the reader read the whole reply for a client that had left"


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_reply_completed_before_the_cancel_is_the_answer_and_is_metered(
    flavor: Flavor, monkeypatch: pytest.MonkeyPatch
):
    turn, usage, _, _ = _cancelled_at_the_stall(flavor, "end", monkeypatch)
    assert isinstance(turn, AssistantTurn)
    assert turn.answer == "hello"
    assert (usage.prompt_tokens, usage.completion_tokens) == (3, 2)


class _LateToken(CancelToken):
    """Reads not-cancelled once, as a token does when the Esc lands just after the check."""

    def __init__(self) -> None:
        super().__init__()
        self.cancel()
        self.checked = False

    @property
    def cancelled(self) -> bool:
        first, self.checked = not self.checked, True
        return False if first else super().cancelled


def test_an_esc_just_after_the_check_sends_no_request():
    client = _Client(_Stream(["hel"], hang=False))
    turn, _ = caller(client, _LateToken())(ASK)
    assert turn == Cancelled(partial="")
    assert client.opened == 1, "the stream is made, and never entered"
    assert not client._stream.closed.is_set()


class _Unclosable(_Stream):
    def close(self) -> None:
        raise ValueError("the stream would not close")


def test_a_stop_whose_close_raises_still_releases_the_op():
    stream, token = _Unclosable(["hel"], hang=True), CancelToken()

    def face() -> None:
        stream.reading.wait(10)
        with pytest.raises(ValueError, match="would not close"):
            token.cancel()

    pressed = threading.Thread(target=face, name="face")
    pressed.start()
    turn, _ = caller(_Client(stream), token)(ASK)
    held = not stream.unblocked.is_set()
    pressed.join()
    stream.release.set()
    assert turn == Cancelled(partial="hel")
    assert held, "the op waited for the read the failed close never ended"


class _Incomplete(_Stream):
    """Ends `incomplete` at the output cap, then holds the connection open."""

    def __iter__(self) -> Any:
        yield SimpleNamespace(type="response.output_text.delta", delta="hel")
        yield SimpleNamespace(type="response.incomplete", response=None)
        self.reading.set()
        self.release.wait(10)
        raise ConnectionError("the connection was reset")


def test_a_reply_that_ended_incomplete_before_the_stop_raises_as_it_would_without_it():
    stream, token = _Incomplete([], hang=True), CancelToken()

    def face() -> None:
        stream.reading.wait(10)
        token.cancel()

    pressed = threading.Thread(target=face, name="face")
    pressed.start()
    try:
        with pytest.raises(RuntimeError, match="without completing"):
            caller(_Client(stream), token)(ASK)
    finally:
        pressed.join()
        stream.release.set()


def test_an_async_call_with_no_token_parses_on_the_loop():
    answered, ran_on = _Stream(["hello"], hang=False).get_final_response(), []

    async def parse(**_: Any) -> Any:
        ran_on.append(asyncio.get_running_loop() is shared_loop().loop)
        return answered

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    turn, usage = AsyncResponsesTurnCaller(
        client=client, system_prompt="s", model="gpt-5.6-luna", price=GPT56_LUNA
    )(ASK)
    assert turn == AssistantTurn(thought="t", answer="hello")
    assert usage.prompt_tokens == 10
    assert ran_on == [True], "the parse ran on the process's one loop"


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_reply_that_ended_incomplete_before_the_stop_is_not_a_cancel(
    flavor: Flavor, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(RuntimeError, match="without completing"):
        _cancelled_at_the_stall(flavor, "ended", monkeypatch)


def test_an_async_call_after_the_cancel_sends_nothing():
    opened: list[dict[str, Any]] = []
    client = SimpleNamespace(
        responses=SimpleNamespace(stream=lambda **kwargs: opened.append(kwargs))
    )
    token = CancelToken()
    token.cancel()
    turn, _ = _caller("async", client, token)(ASK)
    assert turn == Cancelled(partial="")
    assert opened == []


def test_a_cancelled_error_nobody_asked_for_is_not_a_stop():
    async def interrupted() -> None:
        raise asyncio.CancelledError

    with pytest.raises(concurrent.futures.CancelledError):
        shared_loop().run(interrupted())


def test_an_error_after_the_stop_is_retrieved_not_logged():
    loop, logged, token = LoopThread(), [], CancelToken()
    loop.loop.set_exception_handler(lambda _, context: logged.append(context))
    started, tasks = threading.Event(), []

    async def unwinding() -> None:
        try:
            tasks.append(asyncio.current_task())
            started.set()
            await asyncio.sleep(10)
        finally:
            raise OSError("the cleanup failed")

    async def ended(task: asyncio.Task[Any]) -> None:
        await asyncio.wait([task])  # its callbacks run in the order added, the retrieval first

    threading.Thread(target=lambda: (started.wait(SAFETY_NET), token.cancel())).start()
    with pytest.raises(Stopped):
        loop.run(unwinding(), token)
    loop.run(ended(tasks.pop()))
    gc.collect()  # an unretrieved exception is logged when its task is collected
    loop.run(asyncio.sleep(0))
    assert logged == []


def test_the_loop_refuses_to_wait_on_itself():
    loop = LoopThread()

    async def nested() -> None:
        loop.run(asyncio.sleep(0))

    raised: list[BaseException] = []

    def call() -> None:
        try:
            loop.run(nested())
        except RuntimeError as refused:
            raised.append(refused)

    caller_thread = threading.Thread(target=call, daemon=True)  # without the refusal, it hangs
    caller_thread.start()
    caller_thread.join(2)
    assert [str(r.args[0]) for r in raised] == ["an interpreter's loop cannot wait on itself"]


@pytest.mark.parametrize("flavor", FLAVORS)
def test_a_reply_capped_with_no_stop_raises_as_either_caller(flavor: Flavor):
    provider = _Provider("capped")
    try:
        with pytest.raises(RuntimeError, match="without completing"):
            _caller(flavor, _client(flavor, provider), CancelToken())(ASK)
    finally:
        provider.server.shutdown()


class _Registered(CancelToken):
    """A token that cancels itself on another thread once a stop is registered on it."""

    @contextmanager
    def on_cancel(self, stop: Callable[[], None]) -> Generator[None]:
        with super().on_cancel(stop):
            threading.Thread(target=self.cancel).start()
            yield


class _ResetAtOnce(_Registered):
    """A token the face resets the instant it cancels: its flag never reads set."""

    @property
    def cancelled(self) -> bool:
        return False


def test_a_stop_is_a_stop_though_the_token_was_reset_at_once():
    token = _ResetAtOnce()
    with pytest.raises(Stopped):
        shared_loop().run(asyncio.sleep(10), token)


def test_a_coroutine_cancelled_before_its_first_step_is_closed_not_abandoned():
    loop, token = LoopThread(), _Registered()
    release = threading.Event()
    loop.loop.call_soon_threadsafe(release.wait, SAFETY_NET)  # busy, so no task starts yet

    async def never() -> None:
        pass

    coroutine = never()
    with pytest.raises(Stopped):
        loop.run(coroutine, token)
    release.set()
    loop.run(asyncio.sleep(0))  # the loop runs what the stop queued before this sleep
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_runs_on_a_loop_of_its_own():  # a forked threaded process is the case
    shared_loop().run(asyncio.sleep(0))
    child = os.fork()
    if child == 0:  # pragma: no cover - the child reports through its exit code
        answered: list[int] = []
        worker = threading.Thread(
            target=lambda: answered.append(shared_loop().run(asyncio.sleep(0, 7))), daemon=True
        )
        worker.start()
        worker.join(3)
        os._exit(0 if answered == [7] else 1)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "the child waited on its parent's loop"
