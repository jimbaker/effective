"""The probes: each runs one question against one port and returns what it measured.

`tests/` pins every value and `compare.py` prints them as a table, so both read the same numbers.
"""

import asyncio
import contextlib
import io
import os
import sqlite3
import tempfile
import tokenize
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fakes import (
    BAD,
    FORKED,
    GOOD,
    PASSAGES,
    Crash,
    World,
    dspy_lm,
    effective_domain,
    langchain_model,
    once,
    pydantic_ai_model,
)

from effective.keys import Key

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
HERE = Path(__file__).parent
QUESTION = "When was the Rosetta Stone found?"


@dataclass(frozen=True)
class Calls:
    """What a probe counted: model requests, search calls, and what the program saw.

    `answers` counts the requests that asked for the answer; it differs from `model` only where
    the model also chooses the tools."""

    model: int
    search: int
    sees: str = ""
    answers: int | None = None

    def cell(self) -> str:
        chose = (
            f" ({self.answers} answering)"
            if self.answers is not None and self.answers != self.model
            else ""
        )
        sees = f", sees {self.sees}" if self.sees else ""
        return f"{self.model} model{chose}, {self.search} search{sees}"


def delta(world: World, run: Callable[[], Any]) -> Calls:
    """The calls `run` makes against `world`."""
    model, answers, search = world.model.calls, world.model.answers, world.search.calls
    seen = run()
    return Calls(
        world.model.calls - model,
        world.search.calls - search,
        seen or "",
        world.model.answers - answers,
    )


def seen(outcome: Any) -> str:
    return type(outcome).__name__


@contextlib.contextmanager
def scratch() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


# --- Effective ------------------------------------------------------------------------------


@contextlib.contextmanager
def lease(seconds: float) -> Iterator[None]:
    """Shorten the SQLite engine's claim lease, so a worker that died is reclaimed at once."""
    from effective.engines import sqlite

    held, sqlite.CLAIM_LEASE_SECONDS = sqlite.CLAIM_LEASE_SECONDS, seconds
    try:
        yield
    finally:
        sqlite.CLAIM_LEASE_SECONDS = held


class EffectiveWorker:
    """One SQLite store and the worker that runs `program` against `world` as a durable task."""

    def __init__(self, db: Path, world: World, program: Callable[[dict[str, Any]], Any]) -> None:
        from effective.engines.sqlite import SqliteApp
        from effective.handlers.durable import DurableHandler

        self.app = SqliteApp(str(db))

        @self.app.register_task("rag")
        def task(params: dict[str, Any], ctx: Any) -> Any:
            handler = DurableHandler(ctx, effective_domain(world))
            return handler.run(lambda: program(params))

    def close(self) -> None:
        self.app.close()


def effective_repair(repairs: int) -> Calls:
    """`repairs=0` is the program as written; `repairs=1` is the author's re-prompt loop."""
    import effective_rag as port

    from effective.channels import Repair

    world = World()
    world.model.script = [BAD, GOOD]

    def program(params: dict[str, Any]) -> Any:
        match (yield from port.rag(QUESTION, repairs=repairs)):
            case Repair() as repair:
                return {"saw": "Repair", "reason": repair.reason}
            case answer:
                return {"saw": seen(answer)}

    with scratch() as directory:
        worker = EffectiveWorker(directory / "rag.db", world, program)
        run = worker.app.spawn("rag", {})
        calls = delta(world, lambda: worker.app.run_until_result(run).result["saw"])
        worker.close()
        return calls


def effective_crash() -> Calls:
    """The workflow dies after `rag` returned and before the task did; a fresh worker resumes."""
    import effective_rag as port

    world, alive = World(), once()

    def program(params: dict[str, Any]) -> Any:
        answer = yield from port.rag(QUESTION)
        if alive():
            raise Crash("the worker died after the model answered")
        return answer

    with scratch() as directory, lease(0.0):
        first = EffectiveWorker(directory / "rag.db", world, program)
        run = first.app.spawn("rag", {})
        with contextlib.suppress(Crash):
            first.app.work_batch()
        first.close()
        assert (world.model.calls, world.search.calls) == (1, 1), "the crash came too early"
        second = EffectiveWorker(directory / "rag.db", world, program)
        resumed = delta(world, lambda: second.app.run_until_result(run).state.value)
        second.close()
        return resumed


def effective_approval() -> int:
    import effective_rag as port

    world = World()
    with scratch() as directory:
        worker = EffectiveWorker(
            directory / "rag.db", world, lambda p: port.reviewed_rag("q1", QUESTION)
        )
        run = worker.app.spawn("rag", {})
        parked_run = worker.app.run_until_result(run)
        assert parked_run is not None
        assert not parked_run.state.terminal, "the review parks the run"
        (parked,) = worker.app.parked()
        worker.app.emit_event(Key.parse(parked.wake_event), True)
        done = worker.app.run_until_result(run)
        worker.close()
        assert done is not None
        assert done.result["citation"] == 0
    return world.notify.calls


def effective_fork() -> Calls:
    import effective_rag as port

    from effective import RecordingHandler
    from effective.fork import OpIndex, fork_at

    recorder = RecordingHandler({"tool:search": PASSAGES, "respond": GOOD})
    recorder.run(lambda: port.rag(QUESTION))
    world = World()
    tail = lambda: fork_at(  # noqa: E731
        lambda: port.rag(QUESTION), recorder.trace, OpIndex(0), FORKED, effective_domain(world)
    )
    return delta(world, lambda: seen(tail().result))


# --- DSPy -----------------------------------------------------------------------------------


@contextlib.contextmanager
def dspy_world(
    world: World, *, cache: bool = False, cache_dir: Path | None = None
) -> Iterator[Any]:
    """Configure DSPy on `world`; with `cache_dir`, a disk-only cache stands for a new process."""
    import dspy
    import dspy_rag as port

    held = dspy.cache
    if cache_dir is not None:
        dspy.configure_cache(
            enable_disk_cache=True, enable_memory_cache=False, disk_cache_dir=str(cache_dir)
        )
    try:
        with dspy.context(lm=dspy_lm(world.model, cache=cache)):
            # A function, so `Module.deepcopy` shares it rather than copying a counter.
            yield port.RAG(lambda question, k: world.search(question, k))
    finally:
        dspy.cache = held


def dspy_repair(wrapper: str) -> Calls:
    import dspy_rag as port

    world = World()
    world.model.script = [BAD, GOOD]
    with dspy_world(world) as rag, contextlib.redirect_stdout(io.StringIO()):
        wrapped = {"refine": port.refined, "best_of_n": port.best_of}[wrapper](rag)
        return delta(world, lambda: seen(wrapped(question=QUESTION)))


def dspy_crash(cache: bool) -> Calls:
    """DSPy keeps no record of a run, so the resume is a rerun of the module."""
    world = World()
    with scratch() as directory:
        with dspy_world(world, cache=cache, cache_dir=directory) as rag:
            rag(question=QUESTION)
        with dspy_world(world, cache=cache, cache_dir=directory) as rag:
            return delta(world, lambda: seen(rag(question=QUESTION)))


def dspy_fork() -> Calls:
    """A rerun with the search changed."""
    world = World()
    with dspy_world(world) as rag:
        rag(question=QUESTION)
        world.search.results = list(FORKED)
        return delta(world, lambda: seen(rag(question=QUESTION)))


# --- Pydantic AI ----------------------------------------------------------------------------


def pydantic_ai_repair() -> Calls:
    import pydantic_ai_rag as port

    world = World()
    world.model.script = [BAD, GOOD]
    deps, model = port.Deps(search=world.search), pydantic_ai_model(world.model)
    return delta(world, lambda: seen(port.agent.run_sync(QUESTION, deps=deps, model=model).output))


def pydantic_ai_crash() -> Calls:
    """The process dies once the answering response arrives; `StepPersistence` on a SQLite store
    holds the snapshots the resume continues from."""
    import pydantic_ai_rag as port
    from pydantic_ai import Agent
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai_harness import StepPersistence
    from pydantic_ai_harness.step_persistence import SqliteStepStore, continue_run

    world = World()
    deps, model = port.Deps(search=world.search), pydantic_ai_model(world.model)

    async def crash(store: SqliteStepStore) -> None:
        persist = StepPersistence(store=store, run_id="base")
        async with port.agent.iter(
            QUESTION, deps=deps, model=model, capabilities=[persist]
        ) as run:
            async for node in run:
                if Agent.is_call_tools_node(node) and any(
                    isinstance(part, ToolCallPart) and part.tool_name == "final_result"
                    for part in node.model_response.parts
                ):
                    raise Crash("the process died after the model answered")

    async def resume(store: SqliteStepStore) -> str:
        history = await continue_run(store, run_id="base")
        persist = StepPersistence(store=store, run_id="resumed")
        result = await port.agent.run(
            message_history=history, deps=deps, model=model, capabilities=[persist]
        )
        return seen(result.output)

    with scratch() as directory:
        with contextlib.suppress(Crash):
            asyncio.run(crash(SqliteStepStore(database=directory / "steps.db")))
        store = SqliteStepStore(database=directory / "steps.db")
        return delta(world, lambda: asyncio.run(resume(store)))


def pydantic_ai_approval() -> int:
    import pydantic_ai_rag as port
    from pydantic_ai import DeferredToolRequests, DeferredToolResults

    world = World()
    deps = port.Deps(search=world.search, notify=world.notify)
    model = pydantic_ai_model(world.model)
    first = port.reviewed.run_sync(QUESTION, deps=deps, model=model)
    assert isinstance(first.output, DeferredToolRequests)
    approvals = DeferredToolResults(
        approvals={c.tool_call_id: True for c in first.output.approvals}
    )
    second = port.reviewed.run_sync(
        message_history=first.all_messages(),
        deferred_tool_results=approvals,
        deps=deps,
        model=model,
    )
    assert isinstance(second.output, port.GenerateAnswer)
    return world.notify.calls


def pydantic_ai_fork() -> Calls:
    """Fork from the snapshot `StepPersistence` saved once the search returned, with the
    search's return replaced. `fork_run` returns only the latest snapshot, which already holds
    the answer, so the probe lists the snapshots and takes the earlier one."""
    import pydantic_ai_rag as port
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    from pydantic_ai_harness import StepPersistence
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    world = World()
    deps, model = port.Deps(search=world.search), pydantic_ai_model(world.model)

    def forked(message: Any) -> Any:
        if not isinstance(message, ModelRequest):
            return message
        parts = [
            ToolReturnPart("search", FORKED, part.tool_call_id)
            if isinstance(part, ToolReturnPart) and part.tool_name == "search"
            else part
            for part in message.parts
        ]
        return ModelRequest(parts=parts)

    async def record(store: SqliteStepStore) -> None:
        persist = StepPersistence(store=store, run_id="b")
        await port.agent.run(QUESTION, deps=deps, model=model, capabilities=[persist])

    async def fork(store: SqliteStepStore) -> str:
        snapshots = await store.list_snapshots(run_id="b")
        searched = next(
            s
            for s in snapshots
            if isinstance(last := s.messages[-1], ModelRequest)
            and any(isinstance(p, ToolReturnPart) and p.tool_name == "search" for p in last.parts)
        )
        history = [forked(message) for message in searched.messages]
        forked_run = port.agent.run(message_history=history, deps=deps, model=model)
        return seen((await forked_run).output)

    with scratch() as directory:
        store = SqliteStepStore(database=directory / "steps.db")
        asyncio.run(record(store))
        return delta(world, lambda: asyncio.run(fork(store)))


# --- LangGraph ------------------------------------------------------------------------------


@contextlib.contextmanager
def saver(directory: Path) -> Iterator[Any]:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    serde = JsonPlusSerializer(allowed_msgpack_modules=[("langgraph_rag", "GenerateAnswer")])
    connection = sqlite3.connect(directory / "graph.db", check_same_thread=False)
    try:
        yield SqliteSaver(connection, serde=serde)
    finally:
        connection.close()


THREAD = {"configurable": {"thread_id": "t1"}}


def langgraph_repair() -> Calls:
    import langgraph_rag as port

    world = World()
    world.model.script = [BAD, GOOD]
    graph = port.build(world.search, langchain_model(world.model), repairs=1)
    return delta(world, lambda: seen(graph.invoke({"question": QUESTION})["response"]))


def langgraph_crash(where: str, functional: bool = False) -> Calls:
    """`where="inside"` dies in the respond unit once the model answered; `where="after"` dies
    once the respond unit's update is streamed, after it completed. A fresh graph on the same
    checkpoint file resumes the thread."""
    import langgraph_rag as port

    world = World()
    build = port.build_functional if functional else port.build
    start = QUESTION if functional else {"question": QUESTION}
    crash_inside = once() if where == "inside" else (lambda: False)
    with scratch() as directory:
        with saver(directory) as checkpointer, contextlib.suppress(Crash):
            graph = build(
                world.search, langchain_model(world.model, crash_inside), checkpointer=checkpointer
            )
            for update in graph.stream(start, THREAD, stream_mode="updates"):
                if "respond" in update:
                    raise Crash("the process died after the respond unit completed")
        with saver(directory) as checkpointer:
            graph = build(world.search, langchain_model(world.model), checkpointer=checkpointer)
            answer = (lambda done: done) if functional else (lambda done: done["response"])
            return delta(world, lambda: seen(answer(graph.invoke(None, THREAD))))


def langgraph_approval() -> int:
    import langgraph_rag as port
    from langgraph.types import Command

    world = World()
    with scratch() as directory, saver(directory) as checkpointer:
        graph = port.build(
            world.search,
            langchain_model(world.model),
            checkpointer=checkpointer,
            notify=world.notify,
        )
        assert "__interrupt__" in graph.invoke({"question": QUESTION}, THREAD)
        done = graph.invoke(Command(resume=True), THREAD)
        assert done["response"].citation == 0
    return world.notify.calls


def langgraph_fork() -> Calls:
    import langgraph_rag as port

    world = World()
    with scratch() as directory, saver(directory) as checkpointer:
        graph = port.build(world.search, langchain_model(world.model), checkpointer=checkpointer)
        graph.invoke({"question": QUESTION}, THREAD)
        searched = next(s for s in graph.get_state_history(THREAD) if s.next == ("respond",))
        fork = graph.update_state(searched.config, {"passages": FORKED}, as_node="search")
        return delta(world, lambda: seen(graph.invoke(None, fork)["response"]))


# --- lines of code --------------------------------------------------------------------------

PORTS = {
    "Effective": "effective_rag.py",
    "DSPy": "dspy_rag.py",
    "Pydantic AI": "pydantic_ai_rag.py",
    "LangGraph": "langgraph_rag.py",
}


def loc(filename: str) -> int:
    """Lines holding code: not blank, not a comment, not a docstring."""
    source = (HERE / filename).read_text()
    lines: set[int] = set()
    previous = tokenize.NEWLINE
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        docstring = token.type == tokenize.STRING and previous in (
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        )
        quiet = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
        if token.type not in (*quiet, tokenize.ENDMARKER) and not docstring:
            lines.update(range(token.start[0], token.end[0] + 1))
        if token.type not in (tokenize.COMMENT, tokenize.NL):
            previous = token.type
    return len(lines)
