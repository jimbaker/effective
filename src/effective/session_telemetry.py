"""A Claude Code session's own telemetry, in Effective's dialect.

Effective meters an agent's KV-cache use (`cost.Usage` → `cache_hit_ratio` → `telemetry.Span`
→ the OTLP span file). A Claude Code session transcript records the *same* quantities per
response: `input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`,
`output_tokens`. So any transcript reads through the same instrument: map each response into a
`Usage`, lift it to a `Span`, and emit it through `otlp_jsonl_sink`, gated offline by
`check_otlp_line`.

A transcript's `effective.cache_hit_ratio` lands in the attribute the live GSM8K bench reports, so
a recorded production trace and a controlled local experiment read on one dial. In a long agent
session the cache ratio is the dominant term in cost.

    python -m effective.session_telemetry <session.jsonl> [--sidecar out.jsonl]
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from effective.cost import Usage
from effective.telemetry import Span, check_otlp_line, otlp_jsonl_sink


@dataclass(frozen=True)
class Price:
    """Per-1M-token USD prices (Claude Opus 4.x list; approximate)."""

    input: float = 15.0
    cache_read: float = 1.50
    cache_write: float = 18.75
    output: float = 75.0


OPUS = Price()


def usage_from_claude_message(u: dict, price: Price = OPUS) -> Usage:
    """One assistant message's usage block → an Effective `Usage`.

    Anthropic reports the *uncached* prompt as ``input_tokens``; the total input is
    ``input_tokens + cache_read + cache_creation``. Setting ``prompt_tokens`` to that
    total makes ``Usage.cache_hit_ratio`` = cache_read / total, the ratio the benches report."""
    inp = int(u.get("input_tokens", 0) or 0)
    read = int(u.get("cache_read_input_tokens", 0) or 0)
    write = int(u.get("cache_creation_input_tokens", 0) or 0)
    out = int(u.get("output_tokens", 0) or 0)
    cost = (
        inp * price.input
        + read * price.cache_read
        + write * price.cache_write
        + out * price.output
    ) / 1_000_000
    return Usage(
        prompt_tokens=inp + read + write,
        completion_tokens=out,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=write,
        cost=cost,
    )


def session_spans(path: Path, *, session_id: str | None = None) -> Iterator[Span]:
    """Each assistant RESPONSE with usage → an LLM `Span` carrying its `Usage` attributes.

    The span's attributes are exactly `Usage.as_attributes()` (gen_ai.* + cost.*), so a
    session trace and a live bench trace are the same shape on the same dial.

    A response, NOT a record, and the difference is a factor of two. Claude Code writes one
    JSONL record per CONTENT BLOCK, so a single API response that produced
    `[thinking, tool_use, tool_use]` is three records — each carrying a COPY of that one
    response's `usage`. Counting per record therefore multiplies every token by the response's
    block count. Measured on a real 1650-line transcript (2026-08-23): 508 usage-bearing records
    for 232 distinct responses, inflating output tokens 743,621 -> 310,257 (2.40x) and cache
    reads 166,073,446 -> 75,414,017 (2.20x).

    The two factors DIFFER, so this is not a scaling that ratios shrug off: the session's
    cache-hit ratio reads 94.47% per record against 95.92% per response. Skewed less than the
    totals, but skewed.

    `requestId` identifies the response (`message.id` is 1:1 with it, 232/232 on that file). A
    record carrying NEITHER is emitted on its own — absence is not evidence that two records
    are the same call, and collapsing them would be the mirror-image error."""
    sid = session_id or path.stem
    iteration = 0
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "assistant":
            continue
        usage_block = record.get("message", {}).get("usage")
        if not usage_block:
            continue
        response_id = record.get("requestId") or record.get("message", {}).get("id")
        if response_id is not None:
            if response_id in seen:
                continue  # another content block of a response already counted
            seen.add(response_id)
        usage = usage_from_claude_message(usage_block)
        # No `key=`: this is the one `Span` constructor outside `traced`, and it reads a
        # foreign transcript offline rather than observing an op in flight. There is no walk
        # above it and so no placement to read — `Span.key` stays `None`, which the join
        # treats as a left-outer miss rather than an orphan.
        yield Span(
            name=f"LM.{iteration}",
            kind="LLM",
            session_id=sid,
            iteration=iteration,
            usage_attributes=usage.as_attributes(),
            fields={
                "model": record.get("message", {}).get("model", ""),
                "ts": record.get("timestamp", ""),
            },
        )
        iteration += 1


def summarize(path: Path) -> tuple[Usage, int]:
    """Aggregate `Usage` over the session plus the message count (the build-session total)."""
    total = Usage()
    n = 0
    for span in session_spans(path):
        # Re-derive Usage from the span's gen_ai.* attributes (the round-trip
        # Usage.as_attributes writes). cache_creation is read too — omitting it
        # silently zeroed cache-write tokens in the aggregate.
        n += 1
        attrs = dict(span.usage_attributes)
        total = total + Usage(
            prompt_tokens=int(attrs["gen_ai.usage.input_tokens"]),
            completion_tokens=int(attrs["gen_ai.usage.output_tokens"]),
            cache_read_input_tokens=int(attrs["gen_ai.usage.cache_read.input_tokens"]),
            cache_creation_input_tokens=int(attrs.get("gen_ai.usage.cache_write.input_tokens", 0)),
            cost=float(attrs["effective.cost.usd"]),
        )
    return total, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Map a Claude Code session into Effective telemetry.")
    ap.add_argument("session", help="path to a Claude Code session .jsonl")
    ap.add_argument("--sidecar", default="/tmp/session_telemetry.jsonl")
    args = ap.parse_args(argv)

    path = Path(args.session)
    sidecar = Path(args.sidecar)
    sidecar.write_text("")
    sink = otlp_jsonl_sink(sidecar)

    n = 0
    for span in session_spans(path):
        sink(span)
        n += 1
    total, _ = summarize(path)

    print(f"session: {path.name}  ({n} LLM turns)")
    print(f"  total input tokens : {total.prompt_tokens:,}")
    print(f"  output tokens      : {total.completion_tokens:,}")
    print(f"  cache-read tokens  : {total.cache_read_input_tokens:,}")
    print(f"  cache_hit_ratio    : {total.cache_hit_ratio}   <- same dial as the GSM8K bench")
    print(f"  est. cost (Opus)   : ${total.cost:,.2f}")

    problems = sum(
        len(check_otlp_line(json.loads(line)))
        for line in sidecar.read_text().splitlines()
        if line.strip()
    )
    verdict = "clean" if not problems else f"{problems} violation(s)"
    print(f"  sidecar -> {sidecar}  ({verdict}, OTLP)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
