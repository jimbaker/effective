"""A content cache answers an op asked before from its store, and asks everything else."""

import json
import os
import subprocess
import sys
from collections import namedtuple
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, create_model, field_validator

from effective import call_tool
from effective.api import ask_llm
from effective.cache import Cache, FileStore, op_digest
from effective.channels import Message
from effective.cost import Contract, MeteredInterpreter, Usage, serve
from effective.domain import AskLLM, CallTool, Judge, WireNoul
from effective.engines.sqlite import SqliteTaskContext
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.layers import compose_domain, domain_layer
from effective.telemetry import Span, traced

FIRST = UUID("019fa000-0000-7000-8000-00000000c001")
SECOND = UUID("019fa000-0000-7000-8000-00000000c002")
SPENT = Usage(prompt_tokens=10, completion_tokens=5, cost=0.001)


class Page(BaseModel):
    url: str
    status: int


class OtherPage(BaseModel):
    url: str
    status: int


@dataclass
class Answerer:
    """A model and a tool that count what they are asked."""

    asked: list[str] = field(default_factory=list)

    def llm(self, op: AskLLM[Any]) -> tuple[Any, Usage]:
        self.asked.append("llm")
        return "an answer", SPENT

    def tools(self, op: CallTool[Any]) -> Any:
        self.asked.append(op.args["url"])
        return Page(url=op.args["url"], status=200)


def _workflow(urls: tuple[str, ...]):
    def run():
        answer = yield from ask_llm("ask", "What is this?", str)
        pages = []
        for url in urls:
            pages.append((yield from call_tool("fetch", {"url": url}, Page)))
        return {"answer": answer, "statuses": [p.status for p in pages]}

    return run


def _run(app, task: UUID, answerer: Answerer, store: FileStore, urls: tuple[str, ...], sink=None):
    layers = [traced(sink, session_id=str(task))] if sink is not None else []
    cache = Cache(store, {AskLLM: "model-a", CallTool: "model-a"})
    interpreter = MeteredInterpreter(
        answerer.llm, answerer.tools, domain_layers=layers, cache=cache
    )
    handler = DurableHandler(SqliteTaskContext(app.conn, task), interpreter, contract=Contract.V1)
    return handler.run(_workflow(urls)), handler.meter


def test_a_second_run_asks_nothing_and_spends_nothing(sqlite_app, tmp_path):
    app, store = sqlite_app(), FileStore(tmp_path / "cache")
    first, second = Answerer(), Answerer()
    out1, meter1 = _run(app, FIRST, first, store, ("a", "b"))
    out2, meter2 = _run(app, SECOND, second, store, ("a", "b"))
    assert (out1, first.asked, meter1) == (out2, ["llm", "a", "b"], SPENT)
    assert (second.asked, meter2) == ([], Usage())


def test_a_changed_argument_asks_only_that_op(sqlite_app, tmp_path):
    app, store = sqlite_app(), FileStore(tmp_path / "cache")
    _run(app, FIRST, Answerer(), store, ("a", "b"))
    second = Answerer()
    _run(app, SECOND, second, store, ("a", "c"))
    assert second.asked == ["c"]


def test_a_hit_is_marked_on_its_span_and_a_miss_is_not(sqlite_app, tmp_path):
    app, store = sqlite_app(), FileStore(tmp_path / "cache")
    spans: list[Span] = []
    _run(app, FIRST, Answerer(), store, ("a",), sink=spans.append)
    _run(app, SECOND, Answerer(), store, ("a", "b"), sink=spans.append)
    assert [s.fields.get("effective.reused", False) for s in spans] == [
        False,
        False,
        True,
        True,
        False,
    ]


def test_a_reused_answer_carries_what_it_cost_when_asked(sqlite_app, tmp_path):
    app, store = sqlite_app(), FileStore(tmp_path / "cache")
    spans: list[Span] = []
    _run(app, FIRST, Answerer(), store, (), sink=spans.append)
    _run(app, SECOND, Answerer(), store, (), sink=spans.append)
    first, reused = spans
    assert "effective.reused.cost" not in first.fields
    assert (
        reused.fields["effective.reused.cost"],
        reused.fields["effective.reused.prompt_tokens"],
    ) == (
        SPENT.cost,
        SPENT.prompt_tokens,
    )


def test_an_unreadable_entry_is_a_miss_and_is_written_again(sqlite_app, tmp_path):
    app, store = sqlite_app(), FileStore(tmp_path / "cache")
    op = CallTool("fetch", Page, {"url": "a"})
    digest = op_digest(op, "model-a")
    store.write(digest, b"not json")
    answerer = Answerer()
    _run(app, FIRST, answerer, store, ("a",))
    assert answerer.asked == ["llm", "a"]
    assert json.loads(store.read(digest) or b"")["result"] == {"url": "a", "status": 200}


def test_an_op_kind_not_named_is_asked_every_time(tmp_path):
    store, answerer = FileStore(tmp_path / "cache"), Answerer()
    interpreter = MeteredInterpreter(
        answerer.llm, answerer.tools, cache=Cache(store, {"fetch": "model-a"})
    )
    for _ in range(2):
        interpreter.run(AskLLM(messages="q", response_schema=str))
        interpreter.run(CallTool("fetch", Page, {"url": "a"}))
    assert answerer.asked == ["llm", "a", "llm"]


ELSEWHERE = create_model("Page", __module__="elsewhere", url=(str, ...), status=(int, ...))
BASE_ASK = AskLLM(messages=[Message(role="user", content="q")], response_schema=str)
BASE_TOOL = CallTool("fetch", Page, {"url": "a"})
NOUL = WireNoul(instructions="same?", criteria={"true": "yes", "false": "no"})
BASE_JUDGE = Judge(state={"x": 1}, questions={"q": NOUL}, response_schema=dict)

CHANGED = {
    "the message": AskLLM(messages=[Message(role="user", content="r")], response_schema=str),
    "the ask's schema": AskLLM(messages=[Message(role="user", content="q")], response_schema=int),
    "the tool": CallTool("get", Page, {"url": "a"}),
    "the arguments": CallTool("fetch", Page, {"url": "b"}),
    "the tool's schema name": CallTool("fetch", OtherPage, {"url": "a"}),
    "the state": Judge(state={"x": 2}, questions={"q": NOUL}, response_schema=dict),
    "the questions": Judge(state={"x": 1}, questions={"r": NOUL}, response_schema=dict),
    "the schema's module": CallTool("fetch", ELSEWHERE, {"url": "a"}),
    "the op kind": AskLLM(messages={"name": "fetch", "args": {"url": "a"}}, response_schema=Page),
}
BASE = {"the schema's module": BASE_TOOL, "the op kind": BASE_TOOL,
        "the message": BASE_ASK, "the ask's schema": BASE_ASK, "the tool": BASE_TOOL,
        "the arguments": BASE_TOOL, "the tool's schema name": BASE_TOOL,
        "the state": BASE_JUDGE, "the questions": BASE_JUDGE}  # fmt: skip


def test_a_digest_changes_with_every_field_it_covers():
    same = [
        what for what, op in CHANGED.items() if op_digest(op, "m") == op_digest(BASE[what], "m")
    ]
    assert same == []
    assert op_digest(BASE_TOOL, "m") != op_digest(BASE_TOOL, "n")


def test_a_schema_that_keeps_its_name_and_changes_its_fields_misses():
    renamed = create_model("Page", __module__=Page.__module__, url=(str, ...))
    assert (renamed.__module__, renamed.__qualname__) == (Page.__module__, Page.__qualname__)
    assert op_digest(CallTool("fetch", renamed, {"url": "a"}), "m") != op_digest(BASE_TOOL, "m")


def test_a_message_cache_hint_is_not_content():
    hinted = AskLLM(messages=[Message(role="user", content="q", cache=True)], response_schema=str)
    assert op_digest(hinted, "m") == op_digest(BASE_ASK, "m")


EQUAL = {
    "ask": (BASE_ASK, AskLLM(messages=[Message(role="user", content="q")], response_schema=str)),
    "tool": (CallTool("f", Page, {"a": 1, "b": 2}), CallTool("f", Page, {"b": 2, "a": 1})),
    "judge": (BASE_JUDGE, Judge(state={"x": 1}, questions={"q": NOUL}, response_schema=dict)),
}


@pytest.mark.parametrize(("one", "other"), EQUAL.values(), ids=list(EQUAL))
def test_two_ops_with_one_content_share_a_digest(one, other):
    assert op_digest(one, "m") == op_digest(other, "m")


# What a review of the content cache reproduced.


def _cached(tmp_path, tools=lambda op: None, llm=lambda op: ("ok", SPENT)) -> MeteredInterpreter:
    kinds = (AskLLM, CallTool, Judge)
    return MeteredInterpreter(
        llm, tools, cache=Cache(FileStore(tmp_path), dict.fromkeys(kinds, "m"))
    )


DISTINCT = {
    "tuple and list": ((1,), [1]),
    "bytes and str": (b"x", "x"),
    "int and str keys": ({1: "x"}, {"1": "x"}),
    "set and list": ({1, 2}, [1, 2]),
    "bool and int": (True, 1),
}


def test_values_that_differ_by_type_never_share_an_answer(tmp_path):
    shared = [
        what
        for what, (one, other) in DISTINCT.items()
        if op_digest(CallTool("f", str, {"x": one}), "m")
        == op_digest(CallTool("f", str, {"x": other}), "m")
    ]
    assert shared == []


class Prompt(BaseModel):
    cache: str


def test_a_models_cache_field_is_content(tmp_path):
    interpreter = _cached(tmp_path, llm=lambda op: (op.messages.cache, SPENT))
    answers = [interpreter.run(AskLLM(Prompt(cache=c), str)) for c in ("first", "second")]
    assert answers == ["first", "second"]


class Items(BaseModel):
    items: tuple[int, ...] | list[int]


class HasKey(BaseModel):
    key: Key


class Converted(BaseModel):
    amount: int

    @field_validator("amount", mode="before")
    @classmethod
    def cents(cls, dollars: int) -> int:
        return dollars * 100


class Entity(BaseModel):
    """Equal by id alone, whatever its amount."""

    id: int
    amount: int

    @field_validator("amount", mode="before")
    @classmethod
    def cents(cls, dollars: int) -> int:
        return dollars * 100

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and other.id == self.id


class Number(IntEnum):
    ONE = 1


class Unsure(BaseModel):
    value: int

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("no answer")


KEPT = {
    "a Key field": (HasKey, HasKey(key=Key.parse("abc"))),
    "None": (type(None), None),
    "a model": (Page, Page(url="a", status=1)),
}
ASKED_AGAIN = {
    "a tuple for object": (object, (1, 2)),
    "a dict for a model": (Page, {"url": "a", "status": 1}),
    "a tuple field read back as a list": (Items, Items(items=(1, 2))),
    "a validator that transforms again": (Converted, Converted(amount=2)),
    "an equality that ignores a field": (Entity, Entity(id=1, amount=2)),
    "an IntEnum for object": (object, Number.ONE),
}


@pytest.mark.parametrize(
    ("schema", "result", "asks"),
    [(*r, 1) for r in KEPT.values()] + [(*r, 2) for r in ASKED_AGAIN.values()],
    ids=[*KEPT, *ASKED_AGAIN],
)
def test_an_answer_is_kept_only_when_it_reads_back_as_itself(tmp_path, schema, result, asks):
    asked: list[CallTool[Any]] = []
    interpreter = _cached(tmp_path, tools=lambda op: asked.append(op) or result)
    op = CallTool("f", schema)
    first, second = interpreter.run(op), interpreter.run(op)
    assert (first is result, second is result or asks == 1, len(asked)) == (True, True, asks)
    assert type(second) is type(result)


def test_an_equality_that_raises_decides_nothing(tmp_path):
    asked: list[CallTool[Any]] = []
    answer = Unsure(value=1)
    interpreter = _cached(tmp_path, tools=lambda op: asked.append(op) or answer)
    op = CallTool("f", Unsure)
    first, second = interpreter.run(op), interpreter.run(op)
    assert (first is answer, type(second), second.value, len(asked)) == (True, Unsure, 1, 1)


class Extra(BaseModel):
    model_config = ConfigDict(extra="allow")


class Names(list):
    pass


Pair = namedtuple("Pair", "a b")


def test_what_the_digest_keeps_apart_and_what_it_refuses():
    def digest(x: Any) -> str:
        return op_digest(CallTool("f", str, {"x": x}), "m")

    assert digest({1}) != digest(frozenset({1}))
    assert digest(Extra(q="first")) != digest(Extra(q="second"))
    for subclassed in (Names([1]), Pair(1, 2)):
        with pytest.raises(TypeError, match="no encoding"):
            digest(subclassed)


def test_the_cache_alone_answers_as_any_interpreter_does(tmp_path):
    base = MeteredInterpreter(lambda op: ("answer", SPENT), lambda op: None)
    cached = Cache(FileStore(tmp_path), {AskLLM: "m"}).over(base)
    assert [cached.run(AskLLM("q", str)) for _ in range(2)] == ["answer", "answer"]


def test_a_cache_under_serve_answers_a_tool_once(tmp_path):
    asked: list[CallTool[Any]] = []
    base = MeteredInterpreter(lambda op: ("ok", SPENT), lambda op: asked.append(op) or 7)
    domain = serve(base=Cache(FileStore(tmp_path), {CallTool: "m"}).over(base))
    op = CallTool("f", int)
    assert domain.run_metered(op) == domain.run_metered(op) == (7, Usage())
    assert len(asked) == 1


def test_a_hit_on_another_op_does_not_mark_this_ops_span(tmp_path):
    cache = Cache(FileStore(tmp_path), {CallTool: "m"})
    base = MeteredInterpreter(lambda op: ("ok", SPENT), lambda op: op.name, cache=cache)
    base.run(CallTool("aux", str))

    @domain_layer
    def auxiliary(op):
        yield CallTool("aux", str)
        return (yield op)

    spans: list[Span] = []
    domain = compose_domain([traced(spans.append), auxiliary], base)
    assert domain.run(CallTool("main", str)) == "main"
    assert "effective.reused" not in spans[-1].fields


@dataclass
class Labels:
    labels: set[str]


DIGEST_IN_A_PROCESS = """
import sys
from dataclasses import dataclass
from effective.cache import op_digest
from effective.domain import CallTool
@dataclass
class Labels:
    labels: set[str]
print(op_digest(CallTool("f", str, {"x": {"aa", "bb"}, "y": Labels({"cc", "dd", "ee"})}), "m"))
"""


def test_a_digest_is_the_same_under_every_hash_seed():
    seen = {
        subprocess.check_output(
            [sys.executable, "-c", DIGEST_IN_A_PROCESS],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "2", "3")
    }
    assert len(seen) == 1


def test_a_template_prompt_has_a_digest_that_keeps_its_conversion():
    name = "q"
    assert op_digest(AskLLM(t"ask {name}", str), "m") != op_digest(
        AskLLM(t"ask {name!r}", str), "m"
    )


def test_a_value_with_no_encoding_is_refused_before_it_is_asked(tmp_path):
    interpreter = _cached(tmp_path, tools=lambda op: pytest.fail("asked"))
    with pytest.raises(TypeError, match="no encoding for object"):
        interpreter.run(CallTool("f", str, {"x": object()}))


class Unwritable:
    def read(self, digest: str) -> bytes | None:
        return None

    def write(self, digest: str, value: bytes) -> None:
        raise PermissionError(digest)


def test_a_store_that_cannot_write_still_returns_the_paid_answer():
    interpreter = MeteredInterpreter(
        lambda op: ("paid", SPENT),
        lambda op: None,
        cache=Cache(Unwritable(), {AskLLM: "m"}),
    )
    assert interpreter.run(AskLLM("q", str)) == "paid"
    assert interpreter.meter == SPENT


def test_a_new_answerer_misses_only_the_ops_it_answers(tmp_path):
    store, asked = FileStore(tmp_path), []

    def interpreter(searcher: str) -> MeteredInterpreter:
        cache = Cache(store, {"search": searcher, CallTool: "http"})
        return MeteredInterpreter(
            lambda op: ("ok", SPENT), lambda op: asked.append(op.name) or op.name, cache=cache
        )

    for searcher in ("haiku", "haiku", "sonnet"):
        for tool in ("search", "fetch"):
            interpreter(searcher).run(CallTool(tool, str))
    assert asked == ["search", "fetch", "search"]
