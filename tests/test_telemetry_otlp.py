"""The span file: OTLP/JSON lines under the OpenTelemetry GenAI semantic conventions."""

import json
from dataclasses import replace
from typing import Any

import pytest

from effective.cost import Usage
from effective.keys import Key, compose_key
from effective.telemetry import (
    CAPTURE_MAX_CHARS,
    GENAI_OPERATIONS,
    OTLP_SCOPE,
    Code,
    Problem,
    Span,
    _any_value,
    _from_any,
    check_otlp_line,
    decode_otlp_line,
    genai_attributes,
    otlp_jsonl_sink,
    otlp_line,
    sidecar_transcripts,
)

NOW = 1_700_000_000_000_000_000


MODEL = Span(
    name="LM.0",
    kind="LLM",
    session_id="run-1",
    duration_ns=250,
    input_messages=[{"role": "user", "content": "hello"}],
    output_messages=[{"role": "assistant", "content": "hi"}],
    usage_attributes=Usage(
        prompt_tokens=100, completion_tokens=7, cache_read_input_tokens=60, cost=0.02
    ).as_attributes(),
    key=compose_key(t"step;tool:a"),
)

TOOL = Span(
    name="tool.lookup.0",
    kind="TOOL",
    tool_name="lookup",
    session_id="run-1",
    duration_ns=40,
    input_messages=[{"role": "user", "content": '{"q": "x"}'}],
    output_messages=[{"role": "tool", "content": "found"}],
)


def _model_span(**changes: Any) -> Span:
    return replace(MODEL, **changes)


def _tool_span(**changes: Any) -> Span:
    return replace(TOOL, **changes)


def _span(line: dict) -> dict:
    return line["resourceSpans"][0]["scopeSpans"][0]["spans"][0]


def test_a_line_is_one_complete_traces_data_object_that_validates():
    line = otlp_line(_model_span(), now_ns=NOW)

    assert check_otlp_line(line) == []
    assert line["resourceSpans"][0]["scopeSpans"][0]["scope"] == {"name": OTLP_SCOPE}
    resource = line["resourceSpans"][0]["resource"]["attributes"]
    assert resource == [{"key": "service.name", "value": {"stringValue": "effective"}}]


@pytest.mark.parametrize(
    ("span", "name", "kind"),
    [
        pytest.param(_model_span(), "chat", 3, id="a model call is a CLIENT span named chat"),
        pytest.param(_tool_span(), "execute_tool lookup", 1, id="a tool call names its tool"),
        pytest.param(
            Span(name="plan", kind="AGENT", session_id="s", agent_name="coder"),
            "invoke_agent coder",
            1,
            id="an agent names itself",
        ),
        pytest.param(
            Span(name="started r1", kind="CHAIN", session_id="s"),
            "started r1",
            1,
            id="an event keeps its rendered message",
        ),
    ],
)
def test_the_span_name_and_kind_follow_the_conventions(span, name, kind):
    raw = _span(otlp_line(span, now_ns=NOW))

    assert (raw["name"], raw["kind"]) == (name, kind)


def test_times_end_at_emission_and_start_one_duration_earlier():
    raw = _span(otlp_line(_model_span(duration_ns=250), now_ns=NOW))

    assert (raw["startTimeUnixNano"], raw["endTimeUnixNano"]) == (str(NOW - 250), str(NOW))


@pytest.mark.parametrize(
    ("status", "error", "expected"),
    [
        pytest.param("OK", None, {"code": 1}, id="ok"),
        pytest.param("ERROR", "boom", {"code": 2, "message": "boom"}, id="error"),
    ],
)
def test_status_carries_the_otlp_code_and_the_error(status, error, expected):
    raw = _span(otlp_line(_model_span(status=status, error=error), now_ns=NOW))

    assert raw["status"] == expected


def test_a_model_span_carries_the_genai_usage_and_message_attributes():
    attrs = genai_attributes(_model_span())

    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.conversation.id"] == "run-1"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 60
    assert attrs["gen_ai.usage.cache_write.input_tokens"] == 0
    assert attrs["effective.cost.usd"] == 0.02
    assert attrs["effective.key"] == "step;tool:a"
    assert json.loads(attrs["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert json.loads(attrs["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "hi"}],
            "finish_reason": "stop",
        }
    ]


def test_a_tool_span_carries_its_call_arguments_and_result_rather_than_messages():
    attrs = genai_attributes(_tool_span())

    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "lookup"
    assert attrs["gen_ai.tool.call.arguments"] == '{"q": "x"}'
    assert attrs["gen_ai.tool.call.result"] == "found"
    assert "gen_ai.input.messages" not in attrs


def test_the_call_site_uses_the_stable_code_attribute_names():
    attrs = genai_attributes(_model_span(code=Code("app.py", 12, "main")))

    assert (
        attrs["code.file.path"],
        attrs["code.line.number"],
        attrs["code.function.name"],
    ) == ("app.py", 12, "main")


def test_every_operation_a_span_can_name_is_one_the_conventions_define():
    assert {"chat", "execute_tool", "invoke_agent"} == GENAI_OPERATIONS


def test_the_capture_guard_caps_content_on_both_message_and_tool_attributes():
    big = "x" * (CAPTURE_MAX_CHARS * 3)
    model = genai_attributes(_model_span(input_messages=[{"role": "user", "content": big}]))
    tool = genai_attributes(_tool_span(output_messages=[{"role": "tool", "content": big}]))

    sent = json.loads(model["gen_ai.input.messages"])[0]["parts"][0]["content"]
    assert len(sent) < len(big)
    assert "chars elided; blake2b:" in sent
    assert len(tool["gen_ai.tool.call.result"]) < len(big)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(7, id="int"),
        pytest.param(-3, id="negative int"),
        pytest.param(0.25, id="float"),
        pytest.param("text", id="str"),
        pytest.param(["a", 1, False], id="list"),
        pytest.param({"k": "v", "n": 2}, id="mapping"),
    ],
)
def test_an_attribute_value_round_trips_through_any_value(value):
    assert _from_any(_any_value(value)) == value


def test_an_int64_travels_as_a_decimal_string_and_a_bool_is_not_an_int():
    assert _any_value(2**62) == {"intValue": str(2**62)}
    assert _any_value(True) == {"boolValue": True}


def test_a_none_attribute_is_absent_rather_than_encoded():
    raw = _span(otlp_line(_model_span(agent_name=None), now_ns=NOW))

    assert "gen_ai.agent.name" not in {a["key"] for a in raw["attributes"]}


def test_decoding_a_line_gives_back_flat_attributes_and_integer_times():
    (row,) = decode_otlp_line(otlp_line(_model_span(), now_ns=NOW))

    assert row["attributes"] == genai_attributes(_model_span())
    assert row["endTimeUnixNano"] - row["startTimeUnixNano"] == 250
    assert row["parentSpanId"] is None


def test_the_sink_appends_one_line_per_span(tmp_path):
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: NOW)
    sink(_model_span())
    sink(_tool_span())

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert [check_otlp_line(line) for line in lines] == [[], []]


def test_the_transcript_reader_reads_the_messages_a_model_span_sent(tmp_path):
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: NOW)
    sink(_model_span(input_messages=[{"role": "system", "content": "be brief"}, {"content": "q"}]))
    sink(_tool_span(key=compose_key(t"step;tool:lookup")))

    assert sidecar_transcripts(path) == {
        "step;tool:a": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "q"},
        ]
    }


def _broken(**changes) -> dict:
    line = otlp_line(_model_span(), now_ns=NOW)
    _span(line).update(changes)
    return line


@pytest.mark.parametrize(
    ("line", "at", "message"),
    [
        pytest.param(
            {"resourceSpans": []}, ("resourceSpans",), "must be a non-empty list", id="empty"
        ),
        pytest.param(
            _broken(traceId="ABC"),
            ("resourceSpans", 0, "scopeSpans", 0, "spans", 0, "traceId"),
            "must be 32 lowercase hex digits",
            id="trace id",
        ),
        pytest.param(
            _broken(spanId="g" * 16),
            ("resourceSpans", 0, "scopeSpans", 0, "spans", 0, "spanId"),
            "must be 16 lowercase hex digits",
            id="span id",
        ),
        pytest.param(
            _broken(kind=9),
            ("resourceSpans", 0, "scopeSpans", 0, "spans", 0, "kind"),
            "must be an OTLP SpanKind value, 0 to 5",
            id="kind",
        ),
        pytest.param(
            _broken(startTimeUnixNano=str(NOW + 1), endTimeUnixNano=str(NOW)),
            ("resourceSpans", 0, "scopeSpans", 0, "spans", 0),
            "the span ends before it starts",
            id="ends before it starts",
        ),
        pytest.param(
            _broken(endTimeUnixNano=NOW),
            ("resourceSpans", 0, "scopeSpans", 0, "spans", 0),
            "start and end times must be decimal strings of nanoseconds",
            id="time not a string",
        ),
        pytest.param(
            _broken(attributes=[{"key": "n", "value": {"intValue": 3}}]),
            ("resourceSpans", 0, "scopeSpans", 0, "spans", 0, "attributes", "n"),
            "intValue must be a decimal string within int64",
            id="int64 as a JSON number",
        ),
        pytest.param(
            _broken(attributes=[{"key": "gen_ai.operation.name", "value": {"stringValue": "x"}}]),
            (
                "resourceSpans",
                0,
                "scopeSpans",
                0,
                "spans",
                0,
                "attributes",
                "gen_ai.operation.name",
            ),
            "must be chat, execute_tool or invoke_agent",
            id="unknown operation",
        ),
    ],
)
def test_the_validator_names_where_a_line_is_not_otlp(line, at, message):
    assert (at, message) in [(problem.at, problem.message) for problem in check_otlp_line(line)]


def test_a_problem_renders_its_path_its_message_and_the_value_found():
    problem = Problem(("resourceSpans", 0, "spans", 1, "spanId"), "must be hex", "zz")

    assert problem.render() == "resourceSpans[0].spans[1].spanId: must be hex (got 'zz')"


def test_the_cli_passes_a_valid_span_file_and_fails_an_invalid_one(tmp_path, capsys):
    from effective.telemetry import main

    good = tmp_path / "good.jsonl"
    otlp_jsonl_sink(good, clock=lambda: NOW)(_model_span())
    assert main([str(good)]) == 0

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"resourceSpans": []}) + "\n")
    assert main([str(bad)]) == 1
    assert "resourceSpans: must be a non-empty list" in capsys.readouterr().out


# --- span identity and placement ------------------------------------------------------------


def test_a_span_id_is_the_run_the_placed_key_and_the_attempt():
    """Two calls of one tool in one turn share a name and a turn, never a placed key. The id is
    stable across a resume, which restarts the turn counter, and distinct per retry attempt."""
    first = _tool_span(key=compose_key(t"step;tool:read"))
    second = _tool_span(key=Key.parse("step;tool:read#2"))
    retried = replace(first, attempt=1)
    resumed = replace(first, iteration=5)

    assert len({first.span_id, second.span_id, retried.span_id}) == 3
    assert resumed.span_id == first.span_id
    assert first.trace_id == second.trace_id == _model_span().trace_id  # one trace per run


def test_traced_gives_two_calls_in_one_turn_and_a_retry_ids_of_their_own():
    from effective.domain import AskLLM, CallTool
    from effective.layers import TransientError, drive_through, retry_domain
    from effective.telemetry import traced

    spans: list[Span] = []
    layer = traced(spans.append, session_id="run-1")

    def base(op):
        match op:
            case AskLLM():
                return ("ok", Usage())
            case CallTool():
                return {"r": 1}

    for path in ("a", "b"):
        drive_through(
            [layer], CallTool(name="read", args={"path": path}, result_schema=dict), base
        )
    failures = [1]

    def flaky(op):
        if failures:
            failures.pop()
            raise TransientError("blip")
        return ("ok", Usage())

    drive_through([retry_domain(2), layer], AskLLM(messages=[], response_schema=str), flaky)

    assert len({span.span_id for span in spans}) == len(spans) == 4


def test_an_error_span_caps_its_message_and_names_its_type():
    from effective.domain import AskLLM
    from effective.layers import drive_through
    from effective.telemetry import traced

    spans: list[Span] = []

    def boom(_op):
        raise ValueError("x" * 500_000)

    with pytest.raises(ValueError, match="xxx"):
        drive_through(
            [traced(spans.append, session_id="s")], AskLLM(messages=[], response_schema=str), boom
        )

    line = json.dumps(otlp_line(spans[0], now_ns=NOW))
    assert len(line) < 10_000
    assert genai_attributes(spans[0])["error.type"] == "ValueError"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("-inf"), id="negative infinity"),
        pytest.param(2**64, id="past int64"),
    ],
)
def test_a_value_json_cannot_hold_as_a_number_is_still_valid_otlp(value):
    line = otlp_line(_model_span(fields={"effective.v": value}), now_ns=NOW)

    json.dumps(line, allow_nan=False)  # no bare NaN or Infinity token
    assert check_otlp_line(line) == []
    (encoded,) = [a["value"] for a in _span(line)["attributes"] if a["key"] == "effective.v"]
    assert "intValue" not in encoded  # an int past int64 cannot travel as one


def test_an_event_names_no_genai_operation_and_is_still_valid():
    line = otlp_line(Span(name="started r1", kind="CHAIN", session_id="s"), now_ns=NOW)

    assert "gen_ai.operation.name" not in genai_attributes(
        Span(name="e", kind="CHAIN", session_id="s")
    )
    assert check_otlp_line(line) == []


def _span_with(**changes) -> dict:
    line = otlp_line(_model_span(), now_ns=NOW)
    _span(line).update(changes)
    return line


def _with_attributes(*pairs: dict) -> dict:
    """A valid line with `pairs` added to its attributes, so only they can be at fault."""
    line = otlp_line(_model_span(), now_ns=NOW)
    _span(line)["attributes"].extend(pairs)
    return line


@pytest.mark.parametrize(
    "line",
    [
        pytest.param({"resourceSpans": [{"scopeSpans": []}]}, id="no scope spans"),
        pytest.param({"resourceSpans": [{"scopeSpans": [{"spans": []}]}]}, id="no spans"),
        pytest.param({"resourceSpans": ["x"]}, id="a resource that is not an object"),
        pytest.param(_span_with(kind=True), id="kind true"),
        pytest.param(_span_with(status={"code": True}), id="status code true"),
        pytest.param(_span_with(status="ok"), id="status not an object"),
        pytest.param(
            _with_attributes({"key": "n", "value": {"intValue": "--5"}}), id="double minus"
        ),
        pytest.param(_with_attributes({"key": "n", "value": {"intValue": "²"}}), id="superscript"),
        pytest.param(
            _with_attributes({"key": "n", "value": {"intValue": str(2**64)}}), id="past int64"
        ),
        pytest.param(
            _with_attributes({"key": "n", "value": {"doubleValue": "nan"}}),
            id="lowercase nan",
        ),
        pytest.param(
            _with_attributes(
                {"key": "n", "value": {"stringValue": "a"}},
                {"key": "n", "value": {"stringValue": "b"}},
            ),
            id="duplicate key",
        ),
        pytest.param(_span_with(attributes=["x"]), id="an attribute that is not an object"),
        pytest.param(_span_with(startTimeUnixNano="²"), id="a unicode digit in a time"),
        pytest.param(_span_with(attributes=[]), id="a model call naming no operation"),
    ],
)
def test_the_validator_reports_rather_than_passes_or_raises(line):
    assert check_otlp_line(line) != []


def test_a_span_written_after_a_torn_line_survives_it(tmp_path):
    from effective.telemetry import sidecar_measurements

    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: NOW)
    sink(_model_span(key=compose_key(t"step;n:0")))
    with path.open("a") as handle:
        handle.write('{"resourceSpans": [{"scopeSp')  # the writer died here
    sink(_model_span(key=compose_key(t"step;n:1")))  # and a resumed run appends

    assert set(sidecar_measurements(path)) == {"step;n:0", "step;n:1"}


def test_an_older_span_file_is_refused_rather_than_read_as_empty(tmp_path):
    from effective.telemetry import NotASpanFile, sidecar_measurements

    path = tmp_path / "spans.jsonl"
    path.write_text(json.dumps({"spanId": "a", "attributes": {"effective.key": "step;a"}}) + "\n")

    with pytest.raises(NotASpanFile):
        sidecar_measurements(path)


def test_the_cli_reports_a_torn_line_rather_than_raising(tmp_path, capsys):
    from effective.telemetry import main

    path = tmp_path / "spans.jsonl"
    path.write_text('{"resourceSpans": [{"scopeSp\n')
    assert main([str(path)]) == 1
    assert "the line is not JSON" in capsys.readouterr().out


# --- the mapping into OTLP is total over the values a span can carry ------------------------

SURROGATE = "\udc80"  # what `os.listdir` makes of an undecodable filename byte


@pytest.mark.parametrize(
    "span",
    [
        pytest.param(_tool_span(tool_name="read" + SURROGATE), id="lone surrogate in a tool name"),
        pytest.param(
            _tool_span(output_messages=[{"role": "tool", "content": "ls: a" + SURROGATE}]),
            id="lone surrogate in tool output",
        ),
        pytest.param(
            _model_span(input_messages=[{"role": "user", "content": "q" + SURROGATE}]),
            id="lone surrogate in a message",
        ),
        pytest.param(
            _model_span(fields={"effective.path" + SURROGATE: "x"}), id="lone surrogate in a key"
        ),
        pytest.param(
            _model_span(status="ERROR", error="no such file: " + SURROGATE),
            id="lone surrogate in an error",
        ),
        pytest.param(_tool_span(tool_name="a\x00b\x1b[31m"), id="NUL and control characters"),
        pytest.param(_tool_span(tool_name="read🧪"), id="an astral-plane character"),
        pytest.param(_tool_span(tool_name=42), id="a tool name that is not a str"),
        pytest.param(_model_span(fields={"effective.raw": b"\xff\x00"}), id="bytes"),
        pytest.param(Span(name="x" * 1_000_000, kind="CHAIN", session_id="s"), id="a 1 MB name"),
    ],
)
def test_an_odd_value_still_maps_to_a_valid_otlp_line(span):
    line = otlp_line(span, now_ns=NOW)

    assert check_otlp_line(line) == []
    # Writable as UTF-8 with no escaping to hide behind: what an OTLP parser decodes.
    encoded = json.dumps(line, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert len(encoded) < 20_000


def test_a_lone_surrogate_arrives_as_its_visible_escape():
    raw = _span(otlp_line(_tool_span(tool_name="read" + SURROGATE), now_ns=NOW))

    assert raw["name"] == "execute_tool read\\udc80"


def test_bytes_map_to_bytes_value_and_round_trip():
    raw = _span(otlp_line(_model_span(fields={"effective.raw": b"\xff\x00"}), now_ns=NOW))
    (value,) = [a["value"] for a in raw["attributes"] if a["key"] == "effective.raw"]

    assert value == {"bytesValue": "/wA="}
    assert _from_any(value) == b"\xff\x00"


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(
            _with_attributes({"key": "n", "value": {"stringValue": "a" + SURROGATE}}),
            id="a lone surrogate in a string value",
        ),
        pytest.param(
            _with_attributes({"key": "n" + SURROGATE, "value": {"stringValue": "a"}}),
            id="a lone surrogate in a key",
        ),
        pytest.param(_span_with(name="chat" + SURROGATE), id="a lone surrogate in the name"),
        pytest.param(
            _span_with(status={"code": 2, "message": SURROGATE}),
            id="a lone surrogate in the status message",
        ),
        pytest.param(
            _with_attributes({"key": "n", "value": {"bytesValue": "not base64!"}}),
            id="bytes that are not base64",
        ),
    ],
)
def test_the_validator_rejects_what_an_otlp_parser_would(line):
    assert check_otlp_line(line) != []


def test_a_lone_surrogate_in_a_message_is_valid_text_once_the_messages_are_parsed():
    """The messages ride as JSON inside a string attribute, and JSON escapes a lone surrogate, so
    the attribute alone looks valid. A consumer parsing the messages must get valid text too."""
    attrs = genai_attributes(
        _model_span(input_messages=[{"role": "user", "content": "q" + SURROGATE}])
    )

    sent = json.loads(attrs["gen_ai.input.messages"])[0]["parts"][0]["content"]
    sent.encode("utf-8")  # raises on a lone surrogate
    assert sent == "q\\udc80"


# --- the capture guard covers every field a span carries ------------------------------------

JPEG = "/9j/" + "A" * 400_000  # what a base64 image looks like in a message part


def _event_span(**values) -> Span:
    from effective.telemetry import event

    spans: list[Span] = []
    blob = values.get("blob")
    event(t"loaded {blob=}", sink=spans.append, session_id="s")
    return spans[0]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("y" * 1_000_000, id="a 1 MB string"),
        pytest.param({"type": "image", "data": JPEG}, id="an image part"),
        pytest.param(
            [{"type": "text", "text": "see"}, {"type": "image", "data": JPEG}], id="parts"
        ),
        pytest.param(b"\xff" * 1_000_000, id="1 MB of bytes"),
        pytest.param(["z" * 1_000] * 1_000, id="a 1 MB list"),
    ],
)
def test_an_event_carrying_bulk_ships_a_small_valid_line_on_both_sinks(value):
    import io

    from effective.telemetry import log_sink

    span = _event_span(blob=value)
    line = otlp_line(span, now_ns=NOW)
    buffer = io.StringIO()
    log_sink(buffer)(span)

    for written in (json.dumps(line), buffer.getvalue()):
        assert len(written) < 12_000
        assert JPEG[:64] not in written  # an image's bytes never leave
    assert check_otlp_line(line) == []


def test_a_capped_field_keeps_its_head_tail_and_hash():
    message = genai_attributes(_event_span(blob="HEAD" + "y" * 1_000_000 + "TAIL"))["message"]

    assert message.startswith("loaded blob='HEAD")
    assert message.endswith("TAIL'")
    assert "chars elided; blake2b:" in message


def test_an_image_field_becomes_its_hashed_marker():
    attrs = genai_attributes(_event_span(blob={"type": "image", "data": JPEG}))

    assert "media: 400004 chars, blake2b:" in attrs["blob"]


def test_large_bytes_become_a_marker_and_small_bytes_travel_exactly():
    large = genai_attributes(_model_span(fields={"effective.raw": b"\xff" * 1_000}))
    small = genai_attributes(_model_span(fields={"effective.raw": b"\xff\x00"}))

    assert large["effective.raw"].startswith("[media: 1000 bytes, blake2b:")
    assert small["effective.raw"] == b"\xff\x00"


def test_small_structure_passes_the_guard_unchanged():
    fields = {"effective.judge.questions": ["CLOSED", "KIND"], "effective.reused.cost": 0.25}

    attrs = genai_attributes(_model_span(fields=fields))

    assert attrs["effective.judge.questions"] == ["CLOSED", "KIND"]
    assert attrs["effective.reused.cost"] == 0.25


def test_a_capped_name_stays_one_line():
    raw = _span(otlp_line(Span(name="x" * 1_000_000, kind="CHAIN", session_id="s"), now_ns=NOW))

    assert "\n" not in raw["name"]
    assert "chars elided; blake2b:" in raw["name"]
