"""Tests for t-string telemetry — the core surface (events, source location, spans)."""

import io
import json
from string.templatelib import Template
from typing import get_args

import pytest

from effective.cost import Usage
from effective.keys import compose_key
from effective.telemetry import (
    _LEVELS,
    _OPERATION,
    Code,
    Kind,
    MixedSessions,
    Severity,
    Span,
    caller,
    check_otlp_line,
    event,
    fields,
    genai_attributes,
    log_sink,
    measurements,
    otlp_jsonl_sink,
    otlp_span,
    render_message,
    sidecar_measurements,
    sidecar_spans,
    sidecar_transcripts,
)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sent(span: Span) -> str:
    """The text of a model span's first input message, as the span file carries it."""
    return json.loads(genai_attributes(span)["gen_ai.input.messages"])[0]["parts"][0]["content"]


def test_render_message_matches_the_fstring():
    run_id = "abc"
    n = 3
    assert render_message(t"started {run_id} with {n} items") == "started abc with 3 items"


def test_render_message_honors_format_spec_and_conversion():
    x = 3.14159
    name = "ab"
    assert render_message(t"{x:.2f} {name!r}") == "3.14 'ab'"


def test_literal_keyed_tables_cover_every_member():
    # _LEVELS/_OPERATION carry real data keyed by Literal members — deliberately
    # NOT derived via get_args (the values are the point). This guard makes a
    # new Severity/Kind member unable to silently miss its mapping.
    assert set(_LEVELS) == set(get_args(Severity.__value__))
    assert set(_OPERATION) == set(get_args(Kind.__value__))


def test_render_message_conversion_composes_with_format_spec():
    # f-string semantics exactly: the spec applies to the CONVERTED string
    # (a conversion must not drop the spec)
    x = 3.14159
    assert render_message(t"{x!r:>12}") == f"{x!r:>12}"
    assert render_message(t"{None!r:>6}") == f"{None!r:>6}"


def test_fields_lifts_only_bare_identifiers():
    run_id = "abc"
    n = 3
    obj = {"k": 1}
    # {obj["k"]} is not an identifier expression -> message only, not a field
    out = fields(t"{run_id=} {n=} value={obj['k']}")
    assert out == {"run_id": "abc", "n": 3}


def test_caller_captures_the_call_site_for_go_backwards():
    code = caller()  # the Sentry key: where in the code did this come from
    assert isinstance(code, Code)
    assert code.filepath.endswith("test_telemetry.py")
    assert code.function == "test_caller_captures_the_call_site_for_go_backwards"
    assert code.lineno > 0


def test_usage_as_attributes_uses_the_genai_names():
    usage = Usage(prompt_tokens=100, completion_tokens=20, cache_read_input_tokens=40, cost=0.01)
    attrs = usage.as_attributes()
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 40
    assert attrs["effective.cost.usd"] == 0.01
    assert attrs["effective.cache_hit_ratio"] == 0.4


def test_span_ids_are_deterministic_and_correctly_sized():
    a = Span(name="LM.0", kind="LLM", session_id="s1", iteration=0)
    b = Span(name="LM.0", kind="LLM", session_id="s1", iteration=0)
    c = Span(name="LM.1", kind="LLM", session_id="s1", iteration=1)
    assert a.span_id == b.span_id  # replay re-emits the same id (idempotent)
    assert a.span_id != c.span_id
    assert len(a.span_id) == 16  # 16-hex spanId
    assert len(a.trace_id) == 32  # 32-hex traceId
    assert a.trace_id == c.trace_id  # one trace per run
    assert Span(name="LM.0", kind="LLM", session_id="s2").trace_id != a.trace_id


def test_log_sink_filters_by_severity_and_carries_source_location():
    buf = io.StringIO()
    sink = log_sink(buf, min_level="warning")
    sink(Span(name="quiet", kind="CHAIN", session_id="s", severity="info"))  # filtered out
    sink(
        Span(
            name="loud",
            kind="CHAIN",
            session_id="s",
            severity="error",
            code=Code("/a/b.py", 7, "g"),
            error="boom",
        )
    )
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert len(lines) == 1  # the info line was below the floor
    assert lines[0]["level"] == "ERROR"
    assert lines[0]["code.line.number"] == 7  # go-backwards rides the log line
    assert lines[0]["error"] == "boom"


def test_event_captures_fields_and_the_call_site():
    buf = io.StringIO()
    sink = log_sink(buf)
    run_id = "r-9"
    n = 4
    span = event(t"started {run_id=} with {n=}", sink=sink, session_id="s")
    assert span.fields["run_id"] == "r-9"
    assert span.fields["n"] == 4
    # the call site is THIS test function, not event()/caller() internals
    assert span.code is not None
    assert span.code.function == "test_event_captures_fields_and_the_call_site"


def test_traced_layer_composes_under_metered_and_emits_a_valid_span(tmp_path):
    # The anti-DSPy thesis: instrumentation is IN THE LAYER LIST, not a global patch.
    from effective.cost import Usage as _Usage
    from effective.cost import metered
    from effective.domain import AskLLM
    from effective.layers import Interpreter, compose_domain
    from effective.telemetry import traced

    class _Base:  # a recorded base interpreter: AskLLM -> (result, usage), no spend
        def run(self, op):
            return ({"answer": "hi"}, _Usage(prompt_tokens=10, completion_tokens=2, cost=0.001))

    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    accrued: list[_Usage] = []
    stack: Interpreter = compose_domain(
        [metered(accrued.append), traced(sink, session_id="s1")], base=_Base()
    )

    op = AskLLM(messages=[{"role": "user", "content": "hello"}], response_schema=dict)
    result = stack.run(op)

    assert result == {"answer": "hi"}  # metered unwrapped (result, usage) -> result
    assert accrued  # the meter still saw the usage
    assert accrued[0].prompt_tokens == 10

    (line,) = _rows(path)
    assert check_otlp_line(line) == []  # the layer's span is valid OTLP
    (row,) = sidecar_spans(path)
    attrs = row["attributes"]
    assert attrs["gen_ai.operation.name"] == "chat"
    assert json.loads(attrs["gen_ai.input.messages"])[0]["parts"][0]["content"] == "hello"
    assert attrs["gen_ai.usage.input_tokens"] == 10  # usage peeked from the tuple
    assert attrs["effective.iteration"] == 0


def test_traced_layer_records_an_error_span_then_reraises(tmp_path):
    import pytest

    from effective.domain import AskLLM
    from effective.layers import compose_domain
    from effective.telemetry import traced

    class _Boom:
        def run(self, op):
            raise RuntimeError("provider down")

    buf = io.StringIO()
    layer = traced(log_sink(buf, min_level="info"), session_id="s1")
    stack = compose_domain([layer], base=_Boom())
    with pytest.raises(RuntimeError, match="provider down"):
        stack.run(AskLLM(messages=[], response_schema=dict))

    line = json.loads(buf.getvalue().splitlines()[0])
    assert line["level"] == "ERROR"  # the failure became an ERROR span
    assert line["status"] == "ERROR"


# --- the capture guard (M0 task 8 + the bench sidecar bloat) -----------------


def test_capture_guard_caps_long_content_with_a_hashed_marker():
    from effective.telemetry import CAPTURE_MAX_CHARS, Span

    long = "x" * 50_000 + "TAIL-SENTINEL"
    span = Span(
        name="LM.0",
        kind="LLM",
        session_id="s",
        input_messages=[{"role": "user", "content": long}],
    )
    content = _sent(span)
    assert len(content) < CAPTURE_MAX_CHARS + 200  # bounded (marker allowance)
    assert content.startswith("xxx")  # head survives (greppable)
    assert content.endswith("TAIL-SENTINEL")  # tail survives
    assert "chars elided; blake2b:" in content  # the proto-reference marker


def test_capture_guard_elides_media_parts_entirely():
    from effective.telemetry import Span

    fake_jpeg = "/9j/" + "A" * 400_000  # the 401KB photo case
    span = Span(
        name="LM.0",
        kind="LLM",
        session_id="s",
        input_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this photo?"},
                    {"type": "image", "data": fake_jpeg, "media_type": "image/jpeg"},
                ],
            }
        ],
    )
    content = _sent(span)
    assert fake_jpeg[:64] not in content  # no image bytes in the span, at all
    assert "media: 400004 chars, blake2b:" in content  # hashed marker instead
    assert "what is in this photo?" in content  # the text part is untouched


def test_capture_guard_leaves_small_content_verbatim():
    from effective.telemetry import Span

    span = Span(
        name="LM.0",
        kind="LLM",
        session_id="s",
        input_messages=[{"role": "user", "content": "hello"}],
    )
    assert _sent(span) == "hello"


def test_a_capped_span_is_still_valid_otlp(tmp_path):
    path = tmp_path / "capped.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    sink(
        Span(
            name="LM.0",
            kind="LLM",
            session_id="s",
            input_messages=[{"role": "user", "content": "y" * 100_000}],
            output_messages=[{"role": "assistant", "content": "ok"}],
        )
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert check_otlp_line(row) == []


def test_traced_times_the_call_and_span_duration_cross_checks_latency_s():
    """`traced` measures its own `yield` (an independent clock) to fill the span's wall-clock
    duration, so the span's duration (traced's clock) cross-checks the caller's `latency_s` (in
    the Usage) — the latency analog of the cost-vs-meter check."""
    from effective.domain import AskLLM
    from effective.layers import drive_through
    from effective.telemetry import traced

    spans: list[Span] = []
    clock = iter([1.0, 1.5]).__next__  # t0=1.0, end=1.5 -> a 0.5s call

    def base(_op):
        return ("answer", Usage(latency_s=0.5, cost=0.01))  # the caller's reported latency

    out = drive_through(
        [traced(spans.append, session_id="s", clock=clock)],
        AskLLM(messages=[], response_schema=str),
        base,
    )
    assert out == ("answer", Usage(latency_s=0.5, cost=0.01))  # observe, never rewrite

    span = spans[0]
    assert span.duration_ns == 500_000_000  # traced's independent wall-clock (0.5s)

    row = otlp_span(span, now_ns=10_000_000_000)
    assert row["startTimeUnixNano"] == str(10_000_000_000 - 500_000_000)  # end - duration

    # THE CROSS-CHECK: two independent clocks on the same call agree.
    assert span.duration_ns / 1e9 == genai_attributes(span)["effective.latency_s"]  # 0.5 == 0.5


def test_the_capture_guard_covers_the_whole_emitted_line():
    """Emit a span with media in its input and its output and grep the whole line for the payload.

    A sink is an egress boundary whatever it is pointed at, so the guard is only "at the source"
    if everything the sink writes reads through it, output messages included."""
    payload = "A" * 400_000
    span = Span(
        name="LM.0",
        kind="LLM",
        session_id="s",
        input_messages=[{"role": "user", "content": [{"type": "image", "data": payload}]}],
        output_messages=[
            {
                "role": "assistant",
                "content": [{"type": "image_url", "image_url": {"url": payload}}],
            }
        ],
    )
    emitted = json.dumps(otlp_span(span, now_ns=0))
    assert payload[:64] not in emitted, "the line carried media bytes"
    assert len(emitted) < 10_000, f"the line is {len(emitted)} bytes; the cap did not apply"


# --- the producer: spans -> the mapping `from_keys(telemetry=…)` takes -------------------------


def _measured_tool(key: Template, *, ns: int) -> Span:
    return Span(name="tool.t.0", kind="TOOL", session_id="s", duration_ns=ns, key=compose_key(key))


def _measured_llm(key: Template, *, ns: int, usd: float) -> Span:
    return Span(
        name="LM.0",
        kind="LLM",
        session_id="s",
        duration_ns=ns,
        usage_attributes=Usage(cost=usd).as_attributes(),
        key=compose_key(key),
    )


def test_measurements_sum_over_the_several_spans_one_op_can_yield():
    """One op can span more than once — each retry attempt does — so a node's figure is what that
    POSITION cost, retries included. Overwriting would silently report only the last attempt."""
    m = measurements(
        [_measured_llm(t"step;a", ns=100, usd=0.01), _measured_llm(t"step;a", ns=250, usd=0.02)]
    )

    assert m == {"step;a": (pytest.approx(0.03), 350)}


def test_a_tool_span_reports_a_duration_and_no_cost():
    """`effective.cost.usd` rides in `Usage.as_attributes()`, which only an `AskLLM` span
    carries. The cost must stay `None` and NOT become `0.0` — `0.0` reads as "this node was free",
    which is the lie `Node.cost` stopped telling when it stopped being a `float`."""
    (cost, duration) = measurements([_measured_tool(t"step;tool:a", ns=42)])["step;tool:a"]

    assert cost is None
    assert duration == 42


def test_a_mixed_key_keeps_the_cost_it_has():
    """A key whose spans are part measured and part not reports the measured part, not `None`."""
    m = measurements([_measured_tool(t"step;a", ns=10), _measured_llm(t"step;a", ns=5, usd=0.25)])

    assert m == {"step;a": (pytest.approx(0.25), 15)}


def test_a_keyless_span_is_dropped_rather_than_counted_somewhere():
    """`key=None` is a real answer — the recording/replay core publishes no placement — and it
    means "this observation has no address", which is a left-outer miss, not an orphan."""
    assert measurements([Span(name="LM.0", kind="LLM", session_id="s", duration_ns=9)]) == {}


def test_the_sidecar_reader_agrees_with_the_live_fold(tmp_path):
    """THE invariant: the two producers are the same measurement by two routes. A dashboard reads
    the sidecar and a benchmark holds the spans; if these ever disagree, one surface is lying."""
    spans = [
        _measured_llm(t"step;a", ns=100, usd=0.01),
        _measured_llm(t"step;a", ns=250, usd=0.02),  # the same op, a second attempt
        _measured_tool(t"step;tool:b", ns=42),
    ]
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    for span in spans:
        sink(span)

    assert sidecar_measurements(path) == measurements(spans)


def test_a_sidecar_row_without_an_address_is_skipped(tmp_path):
    """Rows written before the span carried a key, or by a keyless minter (`session_telemetry`),
    have nothing to join to and must not be counted under a placeholder."""
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    sink(Span(name="LM.0", kind="LLM", session_id="s", duration_ns=9))  # no key
    sink(_measured_tool(t"step;tool:a", ns=42))

    assert sidecar_measurements(path) == {"step;tool:a": (None, 42)}


def test_two_runs_in_one_sidecar_do_not_silently_sum(tmp_path):
    """A sidecar is append-only and OUTLIVES the run — the one axis on which the two producers
    genuinely differ, and the one nothing pinned.

    `measurements` takes a caller-scoped list, so it cannot see another run. `sidecar_measurements`
    takes a FILE, and the sink appends to it forever. Unique run ids do not help, since checkpoint
    keys are not task-scoped and two runs of one workflow address the same node by construction.

    Asserted as a REFUSAL rather than as a corrected sum, deliberately. Summing the wrong runs and
    summing the right ones are indistinguishable from inside the file; the only honest answer to
    "which run is this?" from an unscoped call is that it cannot tell. That is the loud form, and
    it also catches two databases both minting `run-1`, where a scoping fix keyed on the id alone
    would still be wrong."""
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    for session, ns in (("run-A", 250_000_000), ("run-B", 4_000_000_000)):
        sink(
            Span(
                name="LM.0",
                kind="LLM",
                session_id=session,
                duration_ns=ns,
                usage_attributes=Usage(cost=0.01).as_attributes(),
                key=compose_key(t"step;tool:a"),
            )
        )

    with pytest.raises(MixedSessions, match="2 runs"):
        sidecar_measurements(path)

    # NAMED, it folds exactly that run — the ergonomics the refusal has to preserve, or callers
    # route around it by concatenating anyway.
    assert sidecar_measurements(path, session_id="run-A") == {
        "step;tool:a": (pytest.approx(0.01), 250_000_000)
    }


def test_a_truncated_trailing_line_is_skipped_rather_than_raising(tmp_path):
    """A sidecar is written by a process this repo deliberately kills at every op, so ending
    mid-line is an ordinary outcome rather than corruption.

    Raising here does not surface a data problem: it takes down whatever is reading, which in
    practice is the dashboard's run page (`dashboard.py` re-reads per request) and `tape-check`.
    The rows already written are still true, so the last partial one is dropped and the rest are
    served.

    This is the test that executes `telemetry.py`'s `except json.JSONDecodeError: continue`."""
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    sink(
        Span(
            name="LM.0",
            kind="LLM",
            session_id="s",
            duration_ns=250,
            usage_attributes=Usage(cost=0.01).as_attributes(),
            input_messages=[{"role": "user", "content": "go"}],
            key=compose_key(t"step;tool:a"),
        )
    )
    with path.open("a") as handle:  # the worker died here
        handle.write('{"resourceSpans": [{"scopeSpans": [{"spans": [{"spanId": "b"')

    assert sidecar_measurements(path) == {"step;tool:a": (pytest.approx(0.01), 250)}
    # The same guard on the other reader, and it must serve the COMPLETE row rather than merely
    # not raising — a reader that swallowed the partial line and the good one alike would pass a
    # not-raises assertion and report nothing.
    assert sidecar_transcripts(path) == {"step;tool:a": [{"role": "user", "content": "go"}]}


def test_sidecar_spans_reads_one_runs_span_rows_and_refuses_an_unnamed_mix(tmp_path):
    """The rows a projection folds, for the run named."""
    path = tmp_path / "spans.jsonl"
    sink = otlp_jsonl_sink(path, clock=lambda: 1_700_000_000_000_000_000)
    for session, tool in (("run-A", "a"), ("run-A", "b"), ("run-B", "c")):
        sink(Span(name="tool", kind="TOOL", session_id=session, tool_name=tool))

    with pytest.raises(MixedSessions, match="2 runs"):
        sidecar_spans(path)
    assert [
        row["attributes"]["gen_ai.tool.name"] for row in sidecar_spans(path, session_id="run-A")
    ] == [
        "a",
        "b",
    ]
    assert sidecar_spans(path, session_id="run-C") == []


def test_sidecar_spans_reads_a_single_run_unnamed(tmp_path):
    path = tmp_path / "spans.jsonl"
    otlp_jsonl_sink(path)(Span(name="tool", kind="TOOL", session_id="only", tool_name="a"))
    assert [row["attributes"]["gen_ai.conversation.id"] for row in sidecar_spans(path)] == ["only"]
