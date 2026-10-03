"""t-string telemetry — spans and structured events as a composable layer.

Telemetry in this substrate is a **layer, not a framework**. `effective.cost.metered`
already *peeks* the `(result, Usage)` stream as a `@domain_layer`; `traced` (below) is its
sibling — it emits one span per op. Instrumentation is therefore **explicit and composable**:
a span is emitted because `traced` is in the `compose_domain([...])` list, never because of a
global monkeypatch. "Where is this instrumented?" is answered by reading the layer list.

Three ideas, kept deliberately small (a tangled framework-coupled tracer is the failure mode):

- **t-string surface.** `event(t"started {run_id=}")` lifts identifier-keyed interpolations to
  structured fields *and* renders the human message, from one expression.
- **go backwards (Sentry).** Every event/span captures its **call site** (`Code`: file, line,
  function) via `sys._getframe`, so you can group by *where in the code* it came from.
- **neutral record.** A `Span` is a plain dataclass; the sink maps it to the wire at the edge,
  as OTLP/JSON lines under the OpenTelemetry GenAI semantic conventions. No OTel SDK in the
  core (DB-free, provider-neutral).
"""

import base64
import binascii
import json
import math
import sys
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import blake2b
from pathlib import Path
from string.templatelib import Interpolation, Template
from typing import Any, Literal, TextIO, assert_never

from pydantic_core import to_jsonable_python

from effective.cost import Usage
from effective.domain import AskLLM, AsksModel, CallTool, DomainOp, Judge
from effective.keys import Key
from effective.layers import DomainLayer, current_placement, domain_layer, reuse_watch

type Conversion = Literal["r", "s", "a"]
"""The conversions PEP 750 admits after `!` in an interpolation — the alphabet `_convert` keys."""

type Severity = Literal["debug", "info", "warning", "error"]
type Kind = Literal["LLM", "TOOL", "CHAIN", "AGENT"]

# A span's kind -> the `gen_ai.operation.name` the GenAI conventions give it. An `event()` is
# not a GenAI operation, so it names none and is a plain span.
_OPERATION: dict[Kind, str | None] = {
    "LLM": "chat",
    "TOOL": "execute_tool",
    "CHAIN": None,
    "AGENT": "invoke_agent",
}

# A vanilla LogRecord's attribute names would be the reserved set for a logging-backed sink;
# here the t-string field filter is simpler — only bare identifiers become fields.


def _rendered_item(item: str | Interpolation) -> str:
    """One rendered piece of a log template: a static passes through; an
    interpolation converts (``!r``/``!s``/``!a``) and then formats, exactly as an
    f-string would (the spec applies to the converted string).

    **Total over `Template.__iter__`'s own union**, which is what makes the escape this carried
    unnecessary rather than merely overdue. PEP 750 fixes the alphabet at `str | Interpolation`,
    so `assert_never` is provable here and a third member would be a type error the day the
    language grew one — where an `isinstance` chain would simply fall through the bottom and
    return the last arm's answer for a shape it had never seen.

    This function dispatches on a PEP 750 type with no connection to any telemetry format, so
    nothing blocks its totality."""
    match item:
        case str():
            return item
        # The CONVERSION arm first, and matched on its shape rather than on `None`. A
        # `conversion=None` arm followed by a bare capture reads the same to a human and does not
        # narrow for `ty`: the capture in the later arm keeps the full `Literal["a","r","s"] |
        # None`, so `_convert` is handed a possible `None`. Binding `str() as conversion` is what
        # makes the narrowing a fact rather than an inference about arm order.
        case Interpolation(value=value, conversion=str() as conversion, format_spec=spec):
            return format(_convert(value, conversion), spec or "")
        case Interpolation(value=value, format_spec=spec):
            return format(value, spec or "")
        case unreachable:
            assert_never(unreachable)


def render_message(template: Template) -> str:
    """Render a PEP 750 ``Template`` the way the equivalent f-string would (the human msg)."""
    return "".join(map(_rendered_item, template))


def _convert(value: Any, conversion: Conversion) -> str:
    """The three conversions PEP 750 admits, keyed by the letter `Interpolation` carries.

    Typed as `Conversion` rather than `str` so the table and the parameter cannot drift: a `str`
    would admit a letter with no entry and fail with a `KeyError` at render time, where the
    alphabet is closed and known at type-check time."""
    return {"r": repr, "s": str, "a": ascii}[conversion](value)


def fields(template: Template) -> dict[str, Any]:
    """Identifier-keyed interpolations as structured fields.

    ``t"{run_id=} {n=}"`` yields ``{"run_id": …, "n": …}``. A non-identifier expression
    (``{result.value}``, ``{a + b}``) renders into the message only — never a field."""
    out: dict[str, Any] = {}
    for item in template:
        # lint: totality(filter) — only identifier-valued interpolations become attributes;
        # strings and non-identifier expressions intentionally contribute no field.
        if isinstance(item, Interpolation) and item.expression.isidentifier():
            out[item.expression] = item.value
    return out


@dataclass(frozen=True)
class Code:
    """The source location an event/span came from — the Sentry 'go backwards' key."""

    filepath: str
    lineno: int
    function: str

    def attributes(self) -> dict[str, Any]:
        """The location under the OpenTelemetry `code.*` names."""
        return {
            "code.file.path": self.filepath,
            "code.line.number": self.lineno,
            "code.function.name": self.function,
        }


def caller(depth: int = 1) -> Code:
    """Capture the call site ``depth`` frames above ``caller`` (1 = its direct caller)."""
    frame = sys._getframe(depth)
    return Code(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_qualname)


def _hex_id(seed: str, size: int) -> str:
    """A deterministic hex id from ``seed`` — same inputs on replay re-emit the same id
    (idempotent overwrite, not a duplicate), so no randomness is needed in the core."""
    return blake2b(seed.encode(), digest_size=size).hexdigest()


def _as_text(content: Any) -> str:
    """A message content rendered as text (a str passes through; anything else is JSON)."""
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


# --- the capture guard --------------------------------------------------------
# Spans carry shape, timing, and attribution UNCONDITIONALLY; bytes only up to a
# cap. Above it, content is elided to head + tail around a marker that carries
# the blake2b of the FULL text. The marker is a proto-reference: against a
# content-addressed artifact store the same hash resolves, and "elided" becomes
# "reconstructable, with perms". Media
# parts (base64 images and friends) never enter a span at all: a 401KB photo
# per span is the failure this guards at the source.

CAPTURE_MAX_CHARS = 2_000
_MEDIA_PART_TYPES = frozenset(
    {"image", "image_url", "input_image", "input_audio", "audio", "media", "document", "file"}
)
_MEDIA_DATA_KEYS = ("data", "image_url", "b64_json", "url")


def _elide_media(part: Any) -> Any:
    """Replace a media message-part's payload with a hashed marker (never bytes)."""
    if not isinstance(part, Mapping) or str(part.get("type", "")) not in _MEDIA_PART_TYPES:
        return part
    payload = next(
        (part[k] for k in _MEDIA_DATA_KEYS if isinstance(part.get(k), str | Mapping)), part
    )
    raw = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return {
        "type": str(part.get("type")),
        "elided": _media_marker(len(raw), "chars", _hex_id(raw, 16)),
    }


def _media_marker(size: int, unit: str, digest: str) -> str:
    """What stands in a span for media it never carries: its size and the hash of its bytes."""
    return f"[media: {size} {unit}, blake2b:{digest}]"


def _wire_text(text: str) -> str:
    """`text` as valid UTF-8. A lone surrogate, which no UTF-8 can carry and which Python makes
    from an undecodable filename or a `\\ud800`-style JSON escape, becomes its visible escape
    rather than a line an OTLP parser rejects."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _without_media(content: Any) -> tuple[Any, bool]:
    """`content` with its media parts replaced by hashed markers, and whether it had any."""
    # lint: totality(coercion) — content is `Any` at this telemetry boundary; a list of parts or
    # a single part is elided and every other value passes through untouched.
    match content:
        case list():
            parts = [_elide_media(part) for part in content]
            return parts, any(new is not old for new, old in zip(parts, content, strict=True))
        case Mapping():
            part = _elide_media(content)
            return part, part is not content
        case _:
            return content, False


def _captured(content: Any, *, joiner: str = "\n") -> str:
    """Content as a span attribute: media parts elided, text capped with a hashed elision marker
    (head + tail stay greppable). `joiner` sets the marker apart; a name passes a space, so a
    capped name stays one line."""
    content, _ = _without_media(content)
    text = _wire_text(_as_text(content))
    if len(text) <= CAPTURE_MAX_CHARS:
        return text
    head = text[: (CAPTURE_MAX_CHARS * 3) // 4]
    tail = text[-(CAPTURE_MAX_CHARS // 4) :]
    elided = len(text) - len(head) - len(tail)
    marker = f"…[{elided} chars elided; blake2b:{_hex_id(text, 16)}]…"
    return f"{head}{joiner}{marker}{joiner}{tail}"


_SMALL_BYTES = 256
"""Bytes up to this size travel exactly, as OTLP `bytesValue`; past it they are media."""


def _capped_field(value: Any) -> Any:
    """A span field as the capture guard lets it leave: text capped, media and large bytes
    replaced by a hashed marker, a large or media-bearing structure rendered and capped. Small
    structure, numbers and booleans pass through unchanged."""
    # lint: totality(coercion) — a field is `Any`, whatever an `event()` interpolated; bulk is
    # normalized to capped text and every other value passes through.
    match value:
        case str():
            return _captured(value)
        case bytes() | bytearray() if len(value) > _SMALL_BYTES:
            return _media_marker(len(value), "bytes", blake2b(value, digest_size=16).hexdigest())
        case tuple():
            return _capped_field(list(value))
        case _ if len(_as_text(value)) > CAPTURE_MAX_CHARS or _without_media(value)[1]:
            return _captured(value)
        case _:
            return value


@dataclass
class Span:
    """A neutral telemetry span. Its ids are deterministic: one trace per run, and a span id from
    the run, the placed key and the attempt, so two calls in one process never share one and a
    replay re-emits the same. An op executed again after a crash counts its attempts afresh, so
    it reuses the first execution's id."""

    name: str
    kind: Kind
    session_id: str
    iteration: int = 0
    duration_ns: int = 0  # wall-clock of the call, measured by `traced` around its `yield`
    parent_span_id: str | None = None
    input_messages: Sequence[Mapping[str, Any]] = ()
    output_messages: Sequence[Mapping[str, Any]] = ()
    usage_attributes: Mapping[str, Any] = field(default_factory=dict)
    status: Literal["OK", "ERROR"] = "OK"
    severity: Severity = "info"
    code: Code | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    # The PLACED key of the op this span observes — scope frames, own key and occurrence
    # together (`layers.current_placement`): the join column to the tape. It is unique per
    # occurrence and stable across a resume, so with `attempt` it identifies the span; one op
    # can span several times, once per retry attempt. `None` is a real answer: the
    # recording/replay core publishes no placement, and a layer under unit test has no walk
    # above it, so a keyless span falls back to its name and turn.
    key: Key | None = None
    attempt: int = 0  # which span of this key (or of this name and turn) this is, from 0

    @property
    def span_id(self) -> str:
        where = self.key.stored() if self.key is not None else f"{self.name}/{self.iteration}"
        return _hex_id(f"{self.session_id}/{where}/{self.attempt}", 8)  # 16 hex

    @property
    def trace_id(self) -> str:
        return _hex_id(self.session_id, 16)  # 32 hex: one trace per run


# --- sinks: a span goes to one or more swappable destinations -----------------

type Sink = Callable[[Span], None]


def multi(*sinks: Sink) -> Sink:
    """Fan a span out to several sinks (e.g. a JSONL eval sidecar + a stderr log)."""

    def emit(span: Span) -> None:
        for sink in sinks:
            sink(span)

    return emit


# --- OTLP: the span file under the OpenTelemetry GenAI conventions -----------
#
# One line per span, each a complete OTLP `TracesData` object in the OTLP/JSON encoding (the OTLP
# file exporter's format), so a stock collector can read the file and forward it anywhere. The
# attribute names follow the GenAI semantic conventions; Effective's own carry `effective.`.

OTLP_SCOPE = "effective.telemetry"
GENAI_OPERATIONS: frozenset[str] = frozenset(op for op in _OPERATION.values() if op is not None)

# OTLP's enum values: SPAN_KIND_INTERNAL = 1, SPAN_KIND_CLIENT = 3; STATUS_CODE_OK = 1, _ERROR = 2.
_OTLP_KIND: dict[Kind, int] = {"LLM": 3, "TOOL": 1, "CHAIN": 1, "AGENT": 1}
_OTLP_STATUS: dict[str, int] = {"OK": 1, "ERROR": 2}


def _genai_messages(messages: Sequence[Mapping[str, Any]], *, output: bool) -> str:
    """Messages as the conventions' `{role, parts}` list, each part through the capture guard.

    JSON-encoded into one string attribute, the form every OTLP consumer accepts."""
    out: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": str(message.get("role", "assistant" if output else "user")),
            "parts": [{"type": "text", "content": _captured(message.get("content", ""))}],
        }
        if output:
            item["finish_reason"] = str(message.get("finish_reason", "stop"))
        out.append(item)
    return json.dumps(out)


def _content_attributes(span: Span) -> dict[str, Any]:
    """What a span carried: a tool call's arguments and result, or a model call's messages."""
    attrs: dict[str, Any] = {}
    match span.kind:
        case "TOOL":
            for name, messages in (
                ("gen_ai.tool.call.arguments", span.input_messages),
                ("gen_ai.tool.call.result", span.output_messages),
            ):
                if messages:
                    attrs[name] = _captured(messages[0].get("content", ""))
        case "LLM" | "CHAIN" | "AGENT":
            if span.input_messages:
                attrs["gen_ai.input.messages"] = _genai_messages(span.input_messages, output=False)
            if span.output_messages:
                attrs["gen_ai.output.messages"] = _genai_messages(
                    span.output_messages, output=True
                )
    return attrs


def genai_attributes(span: Span) -> dict[str, Any]:
    """A span's attributes under the GenAI semantic conventions."""
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": _OPERATION[span.kind],
        "gen_ai.conversation.id": span.session_id,
        "effective.iteration": span.iteration,
        "error.type": span.error_type,
    }
    if span.agent_name is not None:
        attrs["gen_ai.agent.name"] = span.agent_name
    if span.tool_name is not None:
        attrs["gen_ai.tool.name"] = span.tool_name
    attrs.update(_content_attributes(span))
    attrs.update(span.usage_attributes)
    if span.key is not None:
        # Flattened ONCE, here at the sink boundary, and never into a name, message or id.
        attrs["effective.key"] = span.key.stored()
    if span.code is not None:
        attrs.update(span.code.attributes())
    attrs.update({name: _capped_field(value) for name, value in span.fields.items()})
    return {name: value for name, value in attrs.items() if value is not None}


def _name_of(span: Span) -> tuple[str, str | None]:
    """The span name the conventions ask for, as structure: the operation, then its target."""
    match span.kind:
        case "LLM":
            return "chat", None
        case "TOOL":
            return "execute_tool", span.tool_name
        case "AGENT":
            return "invoke_agent", span.agent_name
        case "CHAIN":
            return span.name, None


def _render_name(operation: str, target: str | None) -> str:
    """The span name as bytes, once `_name_of` has decided its parts."""
    return operation if target is None else f"{operation} {target}"


_INT64 = range(-(2**63), 2**63)
_NON_FINITE: dict[str, str] = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}


def _any_value(value: Any) -> dict[str, Any]:
    """A Python value as an OTLP `AnyValue`; int64 travels as a decimal string, per OTLP/JSON."""
    match value:
        case bool():
            return {"boolValue": value}
        case int() if value in _INT64:
            return {"intValue": str(value)}
        case int():
            return {"stringValue": str(value)}  # past int64, which OTLP cannot carry as an int
        case float() if math.isfinite(value):
            return {"doubleValue": value}
        case float():
            return {"doubleValue": _NON_FINITE[str(value)]}  # OTLP/JSON quotes these
        case str():
            return {"stringValue": _wire_text(value)}
        case bytes() | bytearray():
            return {"bytesValue": base64.b64encode(value).decode("ascii")}
        case list() | tuple():
            return {"arrayValue": {"values": [_any_value(v) for v in value]}}
        case Mapping():
            return {"kvlistValue": {"values": _key_values(value)}}
        case _:
            return {"stringValue": _wire_text(str(value))}


def _key_values(attrs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Attributes as an OTLP `KeyValue` list. A `None` is absent rather than encoded."""
    return [
        {"key": _wire_text(str(k)), "value": _any_value(v)}
        for k, v in attrs.items()
        if v is not None
    ]


def _from_any(value: Mapping[str, Any]) -> Any:
    """An OTLP `AnyValue` back to a Python value: the inverse of `_any_value`."""
    match value:
        case {"boolValue": bool() as b}:
            return b
        case {"intValue": raw}:
            return int(raw)
        case {"doubleValue": raw}:
            return float(raw)
        case {"stringValue": str() as s}:
            return s
        case {"bytesValue": str() as encoded}:
            return base64.b64decode(encoded)
        case {"arrayValue": {"values": list() as items}}:
            return [_from_any(v) for v in items]
        case {"arrayValue": _}:
            return []
        case {"kvlistValue": {"values": list() as pairs}}:
            return {p["key"]: _from_any(p.get("value", {})) for p in pairs}
        case {"kvlistValue": _}:
            return {}
        case _:
            return None


def otlp_span(span: Span, *, now_ns: int) -> dict[str, Any]:
    """One span in the OTLP/JSON encoding. It ends at `now_ns` and starts its duration earlier."""
    row: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": _captured(_render_name(*_name_of(span)), joiner=" "),
        "kind": _OTLP_KIND[span.kind],
        "startTimeUnixNano": str(now_ns - span.duration_ns),
        "endTimeUnixNano": str(now_ns),
        "attributes": _key_values(genai_attributes(span)),
        "status": {
            "code": _OTLP_STATUS[span.status],
            **({"message": _captured(span.error)} if span.error else {}),
        },
    }
    if span.parent_span_id is not None:
        row["parentSpanId"] = span.parent_span_id
    return row


def otlp_line(span: Span, *, now_ns: int, service_name: str = "effective") -> dict[str, Any]:
    """A span as one complete OTLP `TracesData` object: one line of the span file."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _key_values({"service.name": service_name})},
                "scopeSpans": [
                    {"scope": {"name": OTLP_SCOPE}, "spans": [otlp_span(span, now_ns=now_ns)]}
                ],
            }
        ]
    }


def otlp_jsonl_sink(
    path: str | Path,
    *,
    service_name: str = "effective",
    clock: Callable[[], int] = time.time_ns,
) -> Sink:
    """Append each span to `path` as one OTLP/JSON line. `clock` is injected so tests get stable
    timestamps."""
    target = Path(path)

    def emit(span: Span) -> None:
        line = json.dumps(otlp_line(span, now_ns=clock(), service_name=service_name), default=str)
        # A writer killed mid-line leaves the file without its final newline. Starting on a fresh
        # line keeps this span out of that fragment, where a reader would drop it with the rest.
        torn = target.exists() and target.stat().st_size > 0 and _last_byte(target) != b"\n"
        with target.open("a") as fh:
            fh.write(("\n" if torn else "") + line + "\n")

    return emit


def _last_byte(path: Path) -> bytes:
    with path.open("rb") as fh:
        fh.seek(-1, 2)
        return fh.read(1)


class NotASpanFile(ValueError):
    """A file whose lines are spans in some other format than OTLP, such as an older sidecar."""


def decode_otlp_line(line: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Every span in one OTLP/JSON line, its attributes a flat dict and its times ints."""
    for resource in line.get("resourceSpans", []):
        for scoped in resource.get("scopeSpans", []):
            for raw in scoped.get("spans", []):
                yield {
                    "traceId": raw.get("traceId"),
                    "spanId": raw.get("spanId"),
                    "parentSpanId": raw.get("parentSpanId") or None,
                    "name": raw.get("name"),
                    "kind": raw.get("kind"),
                    "startTimeUnixNano": int(raw.get("startTimeUnixNano", 0)),
                    "endTimeUnixNano": int(raw.get("endTimeUnixNano", 0)),
                    "attributes": {
                        a["key"]: _from_any(a.get("value", {})) for a in raw.get("attributes", [])
                    },
                    "status": raw.get("status", {}),
                }


type Where = tuple[str | int, ...]
"""A location inside one line of the span file, as the path of keys and indexes that reach it."""


@dataclass(frozen=True)
class Problem:
    """One way a line of the span file fails to be OTLP: where, what, and the value found."""

    at: Where
    message: str
    got: Any = None

    def render(self) -> str:
        """The problem as one line of text: the path, the message, then the value found."""
        steps: list[str] = []
        for part in self.at:
            match part:
                case int():
                    steps.append(f"[{part}]")
                case str():
                    steps.append(f".{part}")
                case unreachable:
                    assert_never(unreachable)
        where = "".join(steps)
        found = "" if self.got is None else f" (got {self.got!r})"
        return f"{where.lstrip('.')}: {self.message}{found}" if where else f"{self.message}{found}"


_HEXDIGITS = frozenset("0123456789abcdef")


def _is_hex(value: Any, width: int) -> bool:
    return isinstance(value, str) and len(value) == width and _HEXDIGITS.issuperset(value)


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_base64(raw: str) -> bool:
    try:
        base64.b64decode(raw, validate=True)
    except binascii.Error:
        return False
    return True


def _is_int64(raw: str) -> bool:
    digits = raw.removeprefix("-")
    return digits.isascii() and digits.isdigit() and int(raw) in _INT64


def _double_problems(raw: Any, at: Where) -> Iterator[Problem]:
    match raw:
        case bool():
            yield Problem(at, "doubleValue must be a number", raw)
        case int() | float() if math.isfinite(raw):
            return
        case str() if raw in _NON_FINITE.values():
            return
        case _:
            yield Problem(at, "doubleValue must be finite, or NaN or Infinity quoted", raw)


def _any_value_problems(value: Any, at: Where) -> Iterator[Problem]:
    if not isinstance(value, Mapping) or len(value) != 1:
        yield Problem(at, "an AnyValue holds exactly one typed field", value)
        return
    match value:
        case {"arrayValue": {"values": list() as items}}:
            for i, item in enumerate(items):
                yield from _any_value_problems(item, (*at, i))
        case {"kvlistValue": {"values": list() as pairs}}:
            yield from _attribute_problems(pairs, at)
        case _:
            yield from _scalar_problems(value, at)


def _scalar_problems(value: Mapping[str, Any], at: Where) -> Iterator[Problem]:
    match value:
        case {"intValue": raw}:
            if not (isinstance(raw, str) and _is_int64(raw)):
                yield Problem(at, "intValue must be a decimal string within int64", raw)
        case {"doubleValue": raw}:
            yield from _double_problems(raw, at)
        case {"stringValue": str() as text}:
            if not _is_utf8(text):
                yield Problem(at, "stringValue must be valid UTF-8, with no lone surrogate", text)
        case {"bytesValue": str() as raw}:
            if not _is_base64(raw):
                yield Problem(at, "bytesValue must be base64", raw)
        case {"boolValue": bool()}:
            return
        case _:
            yield Problem(at, "not an OTLP AnyValue", value)


def _attribute_problems(attributes: Any, at: Where) -> Iterator[Problem]:
    """Problems in a `KeyValue` list: its shape, each value, and keys used twice."""
    if not isinstance(attributes, list):
        yield Problem(at, "attributes must be a list of key-value pairs", attributes)
        return
    seen: set[str] = set()
    for i, pair in enumerate(attributes):
        match pair:
            case {"key": str() as key, "value": value}:
                if not _is_utf8(key):
                    yield Problem((*at, i), "a key must be valid UTF-8", key)
                if key in seen:
                    yield Problem((*at, key), "a key appears more than once")
                seen.add(key)
                yield from _any_value_problems(value, (*at, key))
            case _:
                yield Problem((*at, i), "a key-value pair needs a string key and a value", pair)


def _id_problems(raw: Mapping[str, Any], at: Where) -> Iterator[Problem]:
    if not _is_hex(raw.get("traceId"), 32):
        yield Problem((*at, "traceId"), "must be 32 lowercase hex digits", raw.get("traceId"))
    if not _is_hex(raw.get("spanId"), 16):
        yield Problem((*at, "spanId"), "must be 16 lowercase hex digits", raw.get("spanId"))
    if (parent := raw.get("parentSpanId")) and not _is_hex(parent, 16):
        yield Problem((*at, "parentSpanId"), "must be 16 lowercase hex digits", parent)


def _enum_problems(raw: Mapping[str, Any], at: Where) -> Iterator[Problem]:
    match raw.get("kind"):
        case int() as kind if not isinstance(kind, bool) and kind in range(6):
            pass
        case kind:
            yield Problem((*at, "kind"), "must be an OTLP SpanKind value, 0 to 5", kind)
    match raw.get("status", {}):
        case {"message": str() as message} if not _is_utf8(message):
            yield Problem((*at, "status", "message"), "must be valid UTF-8", message)
    match raw.get("status", {}):
        case {"code": int() as code} if not isinstance(code, bool) and code in range(3):
            pass
        case {"code": code}:
            yield Problem(
                (*at, "status", "code"), "must be an OTLP StatusCode value, 0 to 2", code
            )
        case {}:
            pass  # a status with no code is UNSET
        case status:
            yield Problem((*at, "status"), "must be an object", status)


def _otlp_span_problems(raw: Mapping[str, Any], at: Where) -> Iterator[Problem]:
    yield from _id_problems(raw, at)
    match raw.get("name"):
        case str() as name if name and _is_utf8(name):
            pass
        case name:
            yield Problem((*at, "name"), "must be a non-empty string of valid UTF-8", name)
    yield from _enum_problems(raw, at)
    match raw.get("startTimeUnixNano"), raw.get("endTimeUnixNano"):
        case (str() as start, str() as end) if _is_int64(start) and _is_int64(end):
            if int(start) > int(end):
                yield Problem(at, "the span ends before it starts")
        case _:
            yield Problem(at, "start and end times must be decimal strings of nanoseconds")
    attributes = raw.get("attributes", [])
    yield from _attribute_problems(attributes, (*at, "attributes"))
    operations = (
        [
            pair.get("value")
            for pair in attributes
            if isinstance(pair, Mapping) and pair.get("key") == "gen_ai.operation.name"
        ]
        if isinstance(attributes, list)
        else []
    )
    match operations:
        case [] if raw.get("kind") != _OTLP_KIND["LLM"]:
            pass  # only a model call's span must name its operation
        case []:
            yield Problem(
                (*at, "attributes", "gen_ai.operation.name"),
                "a model call must name its operation",
            )
        case [{"stringValue": str() as op}] if op in GENAI_OPERATIONS:
            pass
        case [*_, found]:
            yield Problem(
                (*at, "attributes", "gen_ai.operation.name"),
                "must be chat, execute_tool or invoke_agent",
                found,
            )


def _items(value: Any) -> list[Any]:
    """`value` if it is a non-empty JSON array, else an empty list."""
    # lint: totality(coercion) — `value` is `Any` parsed from a line of JSON; a non-empty array
    # passes through and every other value is normalized to an empty list.
    match value:
        case list() if value:
            return value
        case _:
            return []


def check_otlp_line(line: Any) -> list[Problem]:
    """How a line of the span file fails to be an OTLP `TracesData` object; empty if it is one."""
    return list(_line_problems(line))


def _object(value: Any) -> Mapping[str, Any]:
    """`value` if it is a JSON object, else an empty one, so a lookup in it finds nothing."""
    # lint: totality(coercion) — `value` is `Any` parsed from a line of JSON; an object passes
    # through and every other value is normalized to an empty object.
    match value:
        case Mapping():
            return value
        case _:
            return {}


def _line_problems(line: Any) -> Iterator[Problem]:
    found = _object(line).get("resourceSpans")
    if not (resources := _items(found)):
        yield Problem(("resourceSpans",), "must be a non-empty list", found)
        return
    for r, resource in enumerate(resources):
        at: Where = ("resourceSpans", r)
        found = _object(resource).get("scopeSpans")
        if not (scoped_list := _items(found)):
            yield Problem((*at, "scopeSpans"), "must be a non-empty list", found)
            continue
        for s, scoped in enumerate(scoped_list):
            found = _object(scoped).get("spans")
            if not (spans := _items(found)):
                yield Problem((*at, "scopeSpans", s, "spans"), "must be a non-empty list", found)
                continue
            for n, raw in enumerate(spans):
                where: Where = (*at, "scopeSpans", s, "spans", n)
                if span := _object(raw):
                    yield from _otlp_span_problems(span, where)
                else:
                    yield Problem(where, "a span must be a non-empty object", raw)


_LEVELS: dict[Severity, int] = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def log_sink(stream: TextIO | None = None, *, min_level: Severity = "info") -> Sink:
    """One JSON object per line (level, msg-fields, code location) — the human/CloudWatch
    surface. Severity-filtered; the `Code` rides along so a line points back at its source."""
    out = stream if stream is not None else sys.stderr
    floor = _LEVELS[min_level]

    def emit(span: Span) -> None:
        if _LEVELS[span.severity] < floor:
            return
        record: dict[str, Any] = {
            "level": span.severity.upper(),
            "name": _captured(span.name, joiner=" "),
            "kind": span.kind,
            "gen_ai.conversation.id": span.session_id,
            "status": span.status,
            **({"error": _captured(span.error)} if span.error else {}),
            **(span.code.attributes() if span.code else {}),
            **{name: _capped_field(value) for name, value in span.fields.items()},
        }
        out.write(json.dumps(record, default=str) + "\n")

    return emit


def _guarded(item: str | Interpolation) -> str | Interpolation:
    match item:
        case str():
            return item
        case Interpolation(
            value=value, expression=expression, conversion=conversion, format_spec=spec
        ):
            return Interpolation(_capped_field(value), expression, conversion, spec)
        case unreachable:
            assert_never(unreachable)


def _rendered_event(template: Template) -> str:
    """An event's message: each interpolation passes the capture guard BEFORE it is formatted, so
    an image or a large value is elided or capped as a value. Capping the rendered string would
    be too late, since its head would be the image's own bytes."""
    return "".join(map(_rendered_item, map(_guarded, template)))


def event(
    template: Template,
    *,
    sink: Sink,
    session_id: str = "",
    kind: Kind = "CHAIN",
    level: Severity = "info",
    iteration: int = 0,
) -> Span:
    """Emit an ad-hoc structured event from a t-string. Identifier-keyed interpolations
    become fields; the call site is captured (go-backwards); the rendered message rides as
    the ``message`` field. Returns the emitted ``Span``."""
    span = Span(
        name=_rendered_event(template),
        kind=kind,
        session_id=session_id,
        iteration=iteration,
        severity=level,
        code=caller(2),  # 2: skip event() itself, land on the caller
        fields={**fields(template), "message": _rendered_event(template)},
    )
    sink(span)
    return span


# --- the join: measurements keyed by the PLACED key ---------------------------

# What `graphview.from_keys(telemetry=…)` takes, so a projection can report what a node COST and
# how long it took. It lives here, beside the sink and its validator, because both forms below
# read the span file's names (`effective.key`, `effective.cost.usd`, the span's start and end),
# and knowledge of a format belongs with the format. Putting it in `graphview` would make that
# module import this one and learn what a `Span` is, against its own header ("a pure function of
# recorded op keys"); here it adds no import edge at all.

type Measurements = dict[str, tuple[float | None, int]]
"""`{placed key: (cost_usd_or_None, duration_ns)}` — the left side of the tape's outer join."""


def _fold_measurements(rows: Iterable[tuple[str, float | None, int]]) -> Measurements:
    """Sum per key, keeping `None` distinct from `0.0`.

    Both halves are load-bearing. **Summing** is right because one op can yield several spans —
    each retry attempt spans separately — so a node's figure is what that POSITION cost, retries
    included. **`None` surviving** is right because a TOOL span carries no `effective.cost.usd` at
    all, and a `0.0` there would read as "this node was free", which is exactly the lie
    `Node.cost` stopped telling when it stopped being a `float`. `graphview.regroup` takes the
    same care at the fold; this is the same arithmetic one level down."""
    cost: dict[str, float] = {}
    duration: dict[str, int] = {}
    for key, usd, ns in rows:
        duration[key] = duration.get(key, 0) + ns
        if usd is not None:
            cost[key] = cost.get(key, 0.0) + usd
    return {key: (cost.get(key), ns) for key, ns in duration.items()}


def measurements(spans: Iterable[Span]) -> Measurements:
    """Fold live spans into the mapping `from_keys(telemetry=…)` takes.

    A span with `key=None` is DROPPED, not counted under some placeholder: the recording/replay
    core publishes no placement and a layer under unit test has no walk above it, so `None` means
    "this observation has no address", which is a left-outer miss rather than an orphan."""
    return _fold_measurements(
        (span.key.stored(), _span_cost(span.usage_attributes), span.duration_ns)
        for span in spans
        if span.key is not None
    )


class MixedSessions(ValueError):
    """A sidecar holding more than one run, read without saying which one is wanted."""


def sidecar_measurements(path: str | Path, *, session_id: str | None = None) -> Measurements:
    """The same fold, read back from a JSONL sidecar — what a dashboard has, since spans reach it
    as a file rather than as objects.

    **A SIDECAR IS APPEND-ONLY AND OUTLIVES THE RUN, so an unscoped read of a multi-run file
    REFUSES.** This is the one axis on which the two producers genuinely differ: `measurements`
    takes a caller-scoped list and cannot see another run, while this takes a FILE that
    `otlp_jsonl_sink` appends to forever. Checkpoint keys are not task-scoped, so two runs of
    one workflow address the same node by construction and a naive fold sums them — measured, a
    250 ms run reported 4250 ms because a 4-second run shared its file.

    Unique run ids would not help, since the keys are not scoped by run. Refusing is the loud form,
    and it also catches two databases both minting `run-1`: the read is scoped or it raises, and
    neither silently sums.

    `session_id` names the run to fold. Omitted, a single-run file still reads — the ordinary case
    stays ergonomic, and a multi-run file raises `MixedSessions` rather than answering."""
    return _fold_measurements(
        (key, cost, ns) for key, cost, ns, _ in _scoped_rows(Path(path), session_id)
    )


def sidecar_spans(path: str | Path, *, session_id: str | None = None) -> list[dict[str, Any]]:
    """The span rows of one run in a sidecar, as the sink wrote them, for a projection to fold.

    `session_id` names the run. Omitted, a single-run file reads and a multi-run file raises
    `MixedSessions`, as `sidecar_measurements` does: rows from two runs share placed keys, and a
    projection that counted them together would report a run that never happened."""
    rows = list(_span_rows(Path(path)))
    if session_id is None:
        if len(sessions := _sessions_in(Path(path))) > 1:
            raise MixedSessions(
                f"{path} holds {len(sessions)} runs ({', '.join(sorted(sessions))}); "
                "pass `session_id=` to name the one you want"
            )
        return rows
    return [row for row in rows if _session_of(row) == session_id]


def _span_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Every span in a span file, decoded, skipping anything unreadable.

    **A truncated final line is skipped, not raised on.** An append-only log written by a process
    this repo deliberately kills at every op ends mid-line as an ordinary outcome, and a dashboard
    that 500s because a worker was interrupted mid-write is reporting the wrong problem — the rows
    already written are still true."""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partially-written trailing line; the rows before it stand
        match row:
            case {"resourceSpans": _}:
                yield from decode_otlp_line(row)
            case {"spanId": _} | {"traceId": _}:
                raise NotASpanFile(
                    f"{path} holds spans in a format older than OTLP; re-run to write it again"
                )
            case _:
                pass


def _session_of(row: Mapping[str, Any]) -> str | None:
    """The run a span row belongs to."""
    found = row.get("attributes", {}).get("gen_ai.conversation.id")
    return None if found is None else str(found)


def _row_duration_ns(row: Mapping[str, Any]) -> int:
    return int(row.get("endTimeUnixNano", 0)) - int(row.get("startTimeUnixNano", 0))


def _is_model_span(attributes: Mapping[str, Any]) -> bool:
    return attributes.get("gen_ai.operation.name") == "chat"


def _sent_messages(attributes: Mapping[str, Any]) -> list[dict[str, str]]:
    """A model span's input messages as `{role, content}`, its text parts joined in order."""
    return [
        {
            "role": str(message.get("role", "")),
            "content": "".join(
                str(part.get("content", ""))
                for part in message.get("parts", [])
                if part.get("type") == "text"
            ),
        }
        for message in json.loads(attributes.get("gen_ai.input.messages", "[]"))
    ]


def _sessions_in(path: Path) -> set[str]:
    """Every run a span file holds, by `gen_ai.conversation.id`."""
    return {session for row in _span_rows(path) if (session := _session_of(row)) is not None}


def _scoped_rows(
    path: Path, session_id: str | None
) -> Iterator[tuple[str, float | None, int, str]]:
    """Addressed span rows for ONE run, or a refusal naming the runs it found."""
    rows = list(_span_rows(path))
    sessions = _sessions_in(path)
    if session_id is None and len(sessions) > 1:
        raise MixedSessions(
            f"{path} holds {len(sessions)} runs ({', '.join(sorted(sessions))}) and checkpoint "
            f"keys are not task-scoped, so folding them together would sum different runs onto "
            f"one node — pass `session_id=` to name the one you want"
        )
    for row in rows:
        attributes = row.get("attributes", {})
        session = _session_of(row) or ""
        if session_id is not None and session != session_id:
            continue
        key = attributes.get("effective.key")
        if key is None:  # emitted before the span carried an address, or by a keyless minter
            continue
        yield key, _span_cost(attributes), _row_duration_ns(row), session


def sidecar_transcripts(
    path: str | Path, *, session_id: str | None = None
) -> dict[str, list[dict[str, str]]]:
    """What each LLM span says was actually SENT, keyed by the placed key.

    **LLM spans only, and that scoping is the whole reason this is safe to build.** The span
    sidecar cannot carry the transcript PROPERTY — a TOOL span records the DOMAIN result while the
    loop feeds back `ToolResult.content`, and the relationship between them is tool-specific, so a
    walk over tool spans would have to re-implement a dispatch table it does not own. That is the
    measured negative result, and it stands.

    It does not forbid this. An `AskLLM` span's `gen_ai.input.messages` is the transcript itself,
    recorded on the wire, with no tool-result unwrapping anywhere near it. So the sidecar can serve
    as an INDEPENDENT WITNESS to what replay reconstructs, which is a different job from being the
    thing walked.

    """
    found: dict[str, list[dict[str, str]]] = {}
    sessions = _sessions_in(Path(path))
    if session_id is None and len(sessions) > 1:
        raise MixedSessions(
            f"{path} holds {len(sessions)} runs ({', '.join(sorted(sessions))}); another run's "
            f"transcript at the same key would be compared against this run's replay — pass "
            f"`session_id=` to name the one you want"
        )
    for row in _span_rows(Path(path)):
        attributes = row.get("attributes", {})
        if session_id is not None and _session_of(row) != session_id:
            continue
        key = attributes.get("effective.key")
        if key is None or not _is_model_span(attributes):
            continue
        messages = _sent_messages(attributes)
        # LAST span wins for a repeated key: one op can span several times (a retry attempt spans
        # separately), and the last attempt is the one whose transcript reached the model.
        found[key] = messages
    return found


def _span_cost(attributes: Mapping[str, Any]) -> float | None:
    """`effective.cost.usd` rides in `Usage.as_attributes()`, which only an `AskLLM` span carries,
    so a `None` here is the ordinary case for a tool call, not a missing measurement."""
    usd = attributes.get("effective.cost.usd")
    return None if usd is None else float(usd)


# --- the telemetry layer: a peeking @domain_layer over the op stream -----------


def _jsonable(value: Any) -> Any:
    """A Pydantic model -> its dict; anything else unchanged (for message content)."""
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


def _coerce_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, list):
        return [
            # lint: totality(coercion) — `messages` is `Any` at this telemetry boundary; mappings
            # are copied and every other value is normalized to a user-message mapping.
            dict(m) if isinstance(m, Mapping) else {"role": "user", "content": _as_text(m)}
            for m in messages
        ]
    return [{"role": "user", "content": _as_text(messages)}]


def _op_span(
    op: DomainOp[Any],
    out: Any,
    *,
    iteration: int,
    session_id: str,
    agent_name: str | None,
    key: Key | None = None,
    error: BaseException | None = None,
    duration_ns: int = 0,
) -> Span:
    """Build a `Span` from a domain op and its result — the one place op->span mapping
    lives (a single authoritative mapping, not attributes scattered across functions)."""
    failed = error is not None
    status: Literal["OK", "ERROR"] = "ERROR" if failed else "OK"
    severity: Severity = "error" if failed else "info"
    err = str(error) if failed else None
    common: dict[str, Any] = {
        "session_id": session_id,
        "iteration": iteration,
        "duration_ns": duration_ns,
        "status": status,
        "severity": severity,
        "error": err,
        "error_type": None if error is None else type(error).__qualname__,
        "agent_name": agent_name,
        "key": key,
    }
    match op:
        case AskLLM(messages=messages):
            return _model_span(_coerce_messages(messages), out, failed, common)
        case Judge(state=state, questions=questions):
            asked = _as_text(to_jsonable_python({"state": state, "questions": questions}))
            return _model_span(
                [{"role": "user", "content": asked}],
                out,
                failed,
                common,
                fields={"effective.judge.questions": sorted(questions)},
            )
        case CallTool(name=tool, args=args):
            return Span(
                name=f"tool.{tool}.{iteration}",
                kind="TOOL",
                tool_name=tool,
                input_messages=[{"role": "user", "content": _as_text(args)}],
                output_messages=[]
                if failed
                else [{"role": "tool", "content": _as_text(_jsonable(out))}],
                fields=_skill_fields(op, out, failed=failed),
                **common,
            )
        case unreachable:
            assert_never(unreachable)


def _model_span(
    asked: list[dict[str, Any]],
    out: Any,
    failed: bool,
    common: dict[str, Any],
    fields: Mapping[str, Any] | None = None,
) -> Span:
    """A model call's span, named for its turn: `out` is the `(result, usage)` a metered base
    returns, and `fields` names what kind of call it was when the name does not."""
    result, usage = out if isinstance(out, tuple) and len(out) == 2 else (out, None)
    as_attrs = getattr(usage, "as_attributes", None)
    output = [] if failed else [{"role": "assistant", "content": _as_text(_jsonable(result))}]
    return Span(
        name=f"LM.{common['iteration']}",
        kind="LLM",
        input_messages=asked,
        output_messages=output,
        usage_attributes=as_attrs() if callable(as_attrs) else {},
        fields=dict(fields or {}),
        **common,
    )


def _skill_fields(op: CallTool[Any], out: Any, *, failed: bool) -> dict[str, Any]:
    """Skill/pin span attribution: the surface that separates a skill being *present* from a
    skill being *consulted*, which a benchmark of skill use otherwise confounds.

    **Attribution is by SHAPE, never by the tool's NAME**, and the two arms below are the two
    shapes: args carrying a ``skill`` string, and a *pin-shaped* result (str ``name`` +
    ``content_hash``). A name is NOMINAL — it is a label somebody chose, and it moves when they
    choose again. A tool that discloses a skill takes a skill argument and hands back a pin
    whatever it is called, so those are the properties to read.

    The asymmetry this rules out is worth naming: a model-facing tool cannot carry a ``:`` under
    provider tool-name grammars, so any namespace test here would hold for the substrate's own
    tools and silently not for a bench's. ``activate_skill`` is exactly that case and qualifies
    through its result.

    A failed or pin-less disclose still carries ``gen_ai.skill.name`` from the args, because the
    error path is part of the verified-use record; ``effective.skill.content_hash`` only when a
    pin came back."""
    fields: dict[str, Any] = {}
    if isinstance(arg_name := op.args.get("skill"), str):
        fields["gen_ai.skill.name"] = arg_name
        fields["effective.skill.pin_event"] = str(op.args.get("event", "activate"))
    if failed:
        return fields
    pin_name, pin_hash = getattr(out, "name", None), getattr(out, "content_hash", None)
    if isinstance(pin_name, str) and isinstance(pin_hash, str):
        fields["gen_ai.skill.name"] = pin_name
        fields["effective.skill.pin_event"] = str(op.args.get("event", "activate"))
        fields["effective.skill.content_hash"] = pin_hash
    return fields


def _reused_fields(bought: Usage) -> dict[str, Any]:
    return {
        "effective.reused": True,
        "effective.reused.cost": bought.cost,
        "effective.reused.prompt_tokens": bought.prompt_tokens,
        "effective.reused.completion_tokens": bought.completion_tokens,
    }


def traced(
    sink: Sink,
    *,
    session_id: str = "",
    agent_name: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> DomainLayer[Any]:
    """The telemetry concern as a `@domain_layer` — the sibling of `cost.metered`.

    Peeks each domain op, emits one span to `sink`, and **returns the result
    unchanged** (observe, never rewrite), so it composes *under* `metered`:
    `compose_domain([metered(accrue), traced(sink)], base)` — `traced` sees the
    `(result, usage)` tuple before `metered` unwraps it. A turn counter advances per
    model call, `AskLLM` or `Judge` (`effective.iteration`). Instrumentation is in the layer
    list, explicitly, and never a global patch.

    `traced` also **times its own `yield`** (`clock`, injected for tests) to fill the span's
    wall-clock duration, an INDEPENDENT measurement from the caller's `latency_s`
    (which rides in the `Usage` attributes). The two agree on a healthy call, so span
    duration vs `latency_s` is a telemetry self-consistency check, the latency analog of
    the cost-vs-meter cross-check.

    Composed under `retry_domain` (`serve(retry_domain(2), traced(sink), base)` — traced
    INNER), `traced` is re-entered per attempt, so it spans **each attempt**: a failed one as
    an ERROR span, the succeeding one as OK, and each span's duration times *its own*
    attempt. Each attempt's span carries its own `attempt` and so its own span id, which is what a
    backend that keys on span ids needs to keep every attempt."""
    turn = [0]
    spanned: dict[str, int] = {}

    def emit(span: Span) -> None:
        where = span.key.stored() if span.key is not None else f"{span.name}/{span.iteration}"
        attempt = spanned.get(where, 0)
        spanned[where] = attempt + 1
        sink(replace(span, attempt=attempt))

    @domain_layer
    def run(op: DomainOp[Any]) -> Generator[DomainOp[Any], Any, Any]:
        iteration = turn[0]
        # One read, BEFORE the yield. The handler publishes the placement around the whole
        # dispatch (`handlers/absurd.py`, one drive loop for both durable engines), and
        # `tests/test_placement_visible_to_domain_layer.py` measures that it is non-`None`
        # going in AND coming out and equal across the yield — so reading here is immune to
        # any question about teardown at the inner dispatch's exit, including on the error
        # path below.
        key = current_placement()
        t0 = clock()
        try:
            with reuse_watch() as reused:
                out = yield op
        except Exception as exc:  # record an ERROR span (with elapsed), then re-raise
            dur_ns = int((clock() - t0) * 1e9)
            emit(
                _op_span(
                    op,
                    None,
                    iteration=iteration,
                    session_id=session_id,
                    agent_name=agent_name,
                    key=key,
                    error=exc,
                    duration_ns=dur_ns,
                )
            )
            raise
        dur_ns = int((clock() - t0) * 1e9)
        span = _op_span(
            op,
            out,
            iteration=iteration,
            session_id=session_id,
            agent_name=agent_name,
            key=key,
            duration_ns=dur_ns,
        )
        # An answer a layer below reused from a store says so, since its duration and usage
        # describe the lookup; what the answer cost when it was asked rides beside them.
        match [bought for answered, bought in reused if answered is op]:
            case [bought, *_]:
                emit(replace(span, fields={**span.fields, **_reused_fields(bought)}))
            case []:
                emit(span)
        match op:
            case AsksModel():
                turn[0] += 1  # a model call is a turn, so its span has an id of its own
            case CallTool():
                pass  # a tool call belongs to the turn that chose it
            case unreachable:
                assert_never(unreachable)
        return out

    return run


def problems_in(line: str) -> list[Problem]:
    """A line's problems as OTLP, a line that is not JSON at all among them."""
    try:
        return check_otlp_line(json.loads(line))
    except json.JSONDecodeError:
        return [Problem((), "the line is not JSON, as a writer killed mid-line leaves it")]


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m effective.telemetry <spans.jsonl> ...`` validates each line of a span
    file as OTLP, offline, and exits 1 on any problem."""
    args = argv if argv is not None else sys.argv[1:]
    total = 0
    for path in args:
        for i, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            for problem in problems_in(line):
                print(f"{path}:{i}: {problem.render()}")
                total += 1
    if total:
        print(f"\n{total} OTLP problem(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
