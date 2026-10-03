"""Reuse by content: a domain op answered once is answered from a store when it is asked again.

Replay serves what one run recorded, found by position. A cache serves what any run stored, found
by the op's content, so a second run of a changed workflow asks only the ops that changed.

`Cache(store, answerers)` wraps the metered base of a `MeteredInterpreter` (its `cache=`)
or any `MeteredDomain` given to `serve`. There every op answers `(result, usage)`, so a model call
and a tool call are read one way. A hit reports `Usage()`: the run spends nothing on it. Its entry
keeps what the answer cost when it was asked, and a `traced` layer puts that on the hit's span as
`effective.reused.*`, so a run can still say what each source's answers cost.

An answer is kept only when reading it back through JSON and the op's schema gives the value the
call returned, the same round trip a checkpoint makes; any other answer is asked every time.

| store         | needs                             |
|---------------|-----------------------------------|
| `FileStore`   | a directory                       |
| a Redis store | the `bridge` extra, not built yet |

A durable attempt that asks, stores, and dies before its checkpoint commits is retried from the
cache: the retry records `Usage()`, so the spend of the attempt that died is in no checkpoint, and
only the cache entry keeps it. Before the cache the retry paid again and the first payment was in
no checkpoint either.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from string.templatelib import Template
from typing import Any, Protocol, assert_never

from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_json, to_jsonable_python

from effective.channels import Message
from effective.cost import MeteredDomain, Usage
from effective.domain import AskLLM, CallTool, DomainOp, Judge
from effective.keys import Key
from effective.layers import mark_reused


class Store(Protocol):
    """Bytes under a digest. A value that cannot be read back is a miss."""

    def read(self, digest: str) -> bytes | None: ...

    def write(self, digest: str, value: bytes) -> None: ...


@dataclass(frozen=True)
class FileStore:
    """One file per digest under `root`, each written whole by a rename."""

    root: Path

    def _path(self, digest: str) -> Path:
        return self.root / digest[-2:] / digest

    def read(self, digest: str) -> bytes | None:
        try:
            return self._path(digest).read_bytes()
        except OSError:
            return None

    def write(self, digest: str, value: bytes) -> None:
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, staged = tempfile.mkstemp(dir=path.parent, prefix=".write-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(value)
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise


def op_digest(op: DomainOp[Any], identity: str) -> str:
    """The op's content with the schema it answers in and the `identity` of what answers it.

    | op         | content                                                               |
    |------------|-----------------------------------------------------------------------|
    | `AskLLM`   | the messages; a `Message`'s `cache` flag is a provider hint, left out |
    | `Judge`    | the state and the questions                                           |
    | `CallTool` | the tool's name and its arguments                                     |

    Every value is encoded with its type, so a tuple and a list, or `1` and `"1"`, differ. A value
    of a type with no encoding raises `TypeError` before anything is asked. `identity` names the
    answerer, a model for one, since no op carries it."""
    match op:
        case AskLLM(messages=messages, response_schema=schema):
            content: Any = _encoded(messages)
        case Judge(state=state, questions=questions, response_schema=schema):
            content = _encoded({"state": state, "questions": questions})
        case CallTool(name=name, args=args, result_schema=schema):
            content = _encoded({"name": name, "args": args})
        case unreachable:
            assert_never(unreachable)
    body = [type(op).__name__, identity, _named(schema), _json_schema(schema), content]
    return "sha256-" + hashlib.sha256(_text(body).encode()).hexdigest()


def _text(encoded: Any) -> str:
    return json.dumps(encoded, separators=(",", ":"))


def _named(kind: Any) -> list[str]:
    return [getattr(kind, "__module__", ""), getattr(kind, "__qualname__", repr(kind))]


_SCALARS: dict[type, Callable[[Any], Any]] = {
    bool: lambda v: v,
    int: lambda v: v,
    float: repr,
    str: lambda v: v,
    bytes: bytes.hex,
}
"""Encoded by exact type, so `True` is not `1` and a `str` subclass is not a `str`."""


def _encoded(value: Any) -> Any:
    """`value` as JSON that keeps its type: a pure function of the value, across processes."""
    if value is None:
        return None
    if (scalar := _SCALARS.get(type(value))) is not None:
        return [type(value).__name__, scalar(value)]
    if type(value) in _CONTAINERS:
        return _container(value)
    match value:
        case Key():
            return ["key", value.stored()]
        case Enum():
            return ["enum", _named(type(value)), _encoded(value.value)]
        case Message(role=role, content=content):
            return ["message", role, content]
        case BaseModel():
            state = {n: getattr(value, n) for n in type(value).model_fields}
            extra = value.model_extra or {}
            return ["model", _named(type(value)), _encoded(state), _encoded(extra)]
        case Template():
            holes = [[_encoded(i.value), i.expression, i.conversion, i.format_spec]
                     for i in value.interpolations]  # fmt: skip
            return ["template", list(value.strings), holes]
        case _ if is_dataclass(value) and not isinstance(value, type):
            state = {f.name: getattr(value, f.name) for f in fields(value)}
            return ["dataclass", _named(type(value)), _encoded(state)]
        case _:
            raise TypeError(f"op_digest has no encoding for {type(value).__qualname__}")


_CONTAINERS = frozenset({list, tuple, dict, set, frozenset})
"""Matched by exact type: a subclass, a namedtuple for one, has state these encodings drop."""


def _container(value: Any) -> Any:
    kind = type(value).__name__
    match value:
        case list() | tuple():
            return [kind, [_encoded(v) for v in value]]
        case dict():
            items = [[_encoded(k), _encoded(v)] for k, v in value.items()]
            return [kind, sorted(items, key=_text)]
        case set() | frozenset():
            return [kind, sorted((_encoded(v) for v in value), key=_text)]
        case _:
            raise TypeError(f"{type(value).__qualname__} is not a container")


def _json_schema(schema: Any) -> Any:
    return json.loads(_text(TypeAdapter(schema).json_schema()))


def _schema_of(op: DomainOp[Any]) -> Any:
    match op:
        case AskLLM(response_schema=schema) | Judge(response_schema=schema):
            return schema
        case CallTool(result_schema=schema):
            return schema
        case unreachable:
            assert_never(unreachable)


def _revived(schema: Any, result: Any) -> Any:
    """A stored result read back in the op's schema, the way a checkpoint is."""
    if schema is object:
        return result
    return TypeAdapter(schema).validate_json(to_json(result))


def keep_all(op: DomainOp[Any], result: Any) -> bool:
    """Every answer may be kept."""
    return True


@dataclass(frozen=True)
class Cache:
    """Answer each op an entry of `answerers` names from `store`, when its digest is there.

    `answerers` maps an op class, or a tool's name, to the identity of what answers it: a model
    for a judgment, a searcher for `search`. The identity enters the op's digest, so a new model
    misses only the ops it answers. A tool's name is looked up before `CallTool`. An op nothing
    names is asked every time: a model call is sometimes meant to be asked again."""

    store: Store
    answerers: Mapping[type | str, str]
    keeps: Callable[[DomainOp[Any], Any], bool] = keep_all
    """Whether an answer may be kept: a source may give an answer it would not give again."""

    def answerer(self, op: DomainOp[Any]) -> str | None:
        match op:
            case CallTool(name=name):
                return self.answerers.get(name, self.answerers.get(CallTool))
            case AskLLM() | Judge():
                return self.answerers.get(type(op))
            case unreachable:
                assert_never(unreachable)

    def over(self, base: MeteredDomain) -> Cached:
        return Cached(self, base)


@dataclass(frozen=True)
class Cached:
    """A `MeteredDomain` answering from a `Cache` before it asks `base`."""

    cache: Cache
    base: MeteredDomain

    def run_metered(self, op: DomainOp[Any]) -> tuple[Any, Usage]:
        if (identity := self.cache.answerer(op)) is None:
            return self.base.run_metered(op)
        schema, digest = _schema_of(op), op_digest(op, identity)
        if (stored := self.cache.store.read(digest)) is not None:
            try:
                kept = json.loads(stored)
                result = _revived(schema, kept["result"])
                bought = TypeAdapter(Usage).validate_python(kept["usage"])
            except Exception:  # an entry that does not read back, for any reason, is a miss
                pass
            else:
                mark_reused(op, bought)
                return result, Usage()
        result, usage = self.base.run_metered(op)
        if self.cache.keeps(op, result) and (entry := _entry(schema, result, usage)) is not None:
            with suppress(OSError):
                self.cache.store.write(digest, entry)
        return result, usage

    def run(self, op: DomainOp[Any]) -> Any:
        """The bare result, as any `DomainInterpreter` answers."""
        return self.run_metered(op)[0]


def _entry(schema: Any, result: Any, usage: Usage) -> bytes | None:
    """What to keep for `result`, or `None` when reading it back would not give `result` exactly.

    Exactly means the same `_encoded` form, type by type and field by field, so no `__eq__` of
    the result's decides it: a validator that transforms again, a tuple read back as a list, an
    `IntEnum` read back as an `int` are each asked every time."""
    try:
        stored = to_jsonable_python(result)
        if _text(_encoded(_revived(schema, stored))) != _text(_encoded(result)):
            return None
        return to_json({"result": stored, "usage": to_jsonable_python(usage)})
    except Exception:  # keeping is optional, and the call it follows was paid for
        return None
