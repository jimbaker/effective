"""The ouroboros: a Claude Code session's usage mapped into Effective telemetry.

Pins the usage mapping (total input = uncached + cache_read + cache_write, so
cache_hit_ratio is read/total) and that the emitted spans are valid OTLP lines, the same
span file the live GSM8K bench writes.
"""

import json

from effective.session_telemetry import session_spans, summarize, usage_from_claude_message
from effective.telemetry import check_otlp_line, otlp_jsonl_sink


def test_usage_mapping_totals_and_ratio():
    u = usage_from_claude_message(
        {
            "input_tokens": 1000,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 1000,
            "output_tokens": 500,
        }
    )
    assert u.prompt_tokens == 10000  # uncached + read + write
    assert u.cache_read_input_tokens == 8000
    assert u.cache_hit_ratio == 0.8
    assert u.cost > 0  # priced


def _write_session(path):
    lines = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
        {
            "type": "assistant",
            "timestamp": "2026-06-14T15:54:24Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 50,
                },
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-06-14T15:54:30Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 1900,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": 50,
                },
            },
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines))
    return path


def test_session_spans_write_valid_otlp_lines(tmp_path):
    session = _write_session(tmp_path / "session.jsonl")
    spans = list(session_spans(session, session_id="s"))
    assert len(spans) == 2  # only the two assistant messages with usage
    assert spans[0].usage_attributes["effective.cache_hit_ratio"] == 0.9

    sidecar = tmp_path / "out.jsonl"
    sink = otlp_jsonl_sink(sidecar)
    for span in spans:
        sink(span)
    for line in sidecar.read_text().splitlines():
        assert check_otlp_line(json.loads(line)) == []


def test_summarize_aggregates(tmp_path):
    session = _write_session(tmp_path / "session.jsonl")
    total, n = summarize(session)
    assert n == 2
    assert total.prompt_tokens == 3000  # (100+900) + (100+1900)
    assert total.cache_read_input_tokens == 2800


def test_summarize_does_not_silently_zero_cache_write(tmp_path):
    """cache_creation (cache-WRITE) tokens must survive the attribute round-trip;
    as_attributes dropped the key, so the aggregate silently read 0."""
    line = {
        "type": "assistant",
        "timestamp": "2026-06-14T15:54:24Z",
        "message": {
            "role": "assistant",
            "model": "claude-opus-4-8",
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 500,  # a cache WRITE
                "output_tokens": 50,
            },
        },
    }
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps(line))
    total, _ = summarize(session)
    assert total.cache_creation_input_tokens == 500  # was: 0 (silently dropped)


def test_one_response_many_content_blocks_is_counted_once(tmp_path):
    """The fan-out. Claude Code writes one record per CONTENT BLOCK, and every record of one
    response carries a COPY of that response's `usage` — so counting per record multiplies each
    token by the block count. Measured on a real 1650-line transcript: 508
    usage-bearing records for 232 responses, output 743,621 per record against 310,257 per
    response (2.40x), cache reads 2.20x. The factors DIFFER, so the session's cache-hit ratio
    moves too (94.47% -> 95.92%) — this is not a scaling that ratios shrug off."""
    usage = {
        "input_tokens": 100,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
        "output_tokens": 50,
    }
    path = tmp_path / "fanout.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "assistant",
                    "requestId": "req_1",  # ONE response...
                    "message": {"id": "msg_1", "role": "assistant", "usage": usage},
                }
            )
            for _ in range(3)  # ...written as three content-block records
        )
    )

    total, n = summarize(path)
    assert n == 1, "three records of one response are one call"
    assert total.completion_tokens == 50, "not 150"


def test_records_without_a_response_id_are_not_collapsed(tmp_path):
    """The mirror-image error. Absence of `requestId` is not evidence that two records are the
    same call, so a record carrying neither id counts on its own."""
    usage = {
        "input_tokens": 100,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 0,
        "output_tokens": 50,
    }
    path = tmp_path / "noid.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"type": "assistant", "message": {"role": "assistant", "usage": usage}})
            for _ in range(3)
        )
    )

    total, n = summarize(path)
    assert n == 3
    assert total.completion_tokens == 150
