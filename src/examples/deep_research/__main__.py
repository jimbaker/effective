"""Research a question on the open web.

    python -m examples.deep_research "Kestrel probe" launch="the year it launched" \\
        operator="the agency that operates it" [--steps N] [--dir DIR] [--held] [--model M]

Each cell is `name="what it asks"`. The run lives in a durable SQLite store at `DIR/runs.db`.
Search is Brave's API, with `BRAVE_SEARCH_API_KEY` from the environment, and every response is
kept in `DIR/search`, so `--held` replays a run's searches with no key and no spend. Pages are
read by `claude -p` on the subscription, and leads are judged by Jev, with `JEV_API_KEY`. Each of
the three spends against its own cap in `DIR`, checked before every call. Reads, judgments and
pages are kept in `DIR/ops` by their content, so a run that asks nothing new spends nothing. It
prints the report.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from typesafe_sdk import TypeSafeClient

from effective.cache import Cache, FileStore
from effective.cost import Contract, MeteredInterpreter
from effective.domain import AskLLM, CallTool, Judge
from effective.handlers.absurd import DurableHandler
from effective.interpreters.cli import ClaudePrint
from effective.interpreters.jev import Jev
from effective.interpreters.web import (
    FETCH_TOOL,
    SEARCH_TOOL,
    BraveSearch,
    HeldSearch,
    fetch,
    lasting,
)
from effective.keys import Key
from effective.markdown import table
from effective.spend import TokenBudget
from effective.sqlite import SqliteApp, TaskSnapshot
from effective.telemetry import otlp_jsonl_sink, sidecar_spans, traced
from examples.deep_research.research import research

SEARCHES = 24
"""Brave requests a run may send: two leads a step, a query each, for the default steps."""
LLM_TOKENS = 1_000_000
JEV_TOKENS = 200_000
SPANS = "spans.jsonl"


def investigate(
    store: Path,
    question: str,
    cells: Mapping[str, str],
    steps: int,
    answer: MeteredInterpreter,
    run_id: str | None = None,
    **policy: Any,
) -> dict[str, Any]:
    """Run the research as one durable task in `store`, in one attempt, and return its report.
    `run_id` names the task, and is the session its spans carry; `policy` is passed to
    `research`."""
    app = SqliteApp(str(store))
    try:

        @app.register_task("research")
        def task(params: dict[str, Any], ctx: Any) -> Any:
            handler = DurableHandler(ctx, answer, contract=Contract.V1)
            return handler.run(
                lambda: research(params["question"], params["cells"], steps=steps, **policy)
            )

        params = {"question": question, "cells": dict(cells)}
        spawned = app.spawn(
            "research", params, idempotency_key=run_id or str(uuid4()), max_attempts=1
        )
        match app.run_until_result(spawned):
            case TaskSnapshot(state="completed", result=report):
                return report
            case other:
                raise RuntimeError(f"the research did not complete: {other!r}")
    finally:
        app.close()


def live(
    root: Path, model: str, held: bool, searches: int = SEARCHES, session: str | None = None
) -> MeteredInterpreter:
    """Brave or the kept searches, the fetcher, `claude -p` and Jev, each under its cap, with
    every read, judgment and page answered once by its content. With a `session`, each op's span
    goes to `root/spans.jsonl` under it."""
    if held:
        searcher: Any = HeldSearch(root / "search")
    else:
        key = os.environ.get("BRAVE_SEARCH_API_KEY") or sys.exit("BRAVE_SEARCH_API_KEY is not set")
        searcher = BraveSearch(
            key, TokenBudget(root / "brave-requests", searches), root / "search"
        )
    jev_key = os.environ.get("JEV_API_KEY") or sys.exit("JEV_API_KEY is not set")
    tools = {SEARCH_TOOL: searcher, FETCH_TOOL: fetch}

    def tool(op: CallTool[Any]) -> Any:
        return tools[op.name](op)

    jev = Jev(TypeSafeClient(api_key=jev_key), TokenBudget(root / "jev-spend", JEV_TOKENS))
    answered: dict[type | str, str] = {AskLLM: model, Judge: jev.model, FETCH_TOOL: "httpx"}
    return MeteredInterpreter(
        llm=ClaudePrint(model, TokenBudget(root / "llm-spend", LLM_TOKENS)),
        tools=tool,
        judge=jev,
        cache=Cache(FileStore(root / "ops"), answered, keeps=lasting),
        domain_layers=[]
        if session is None
        else [traced(otlp_jsonl_sink(root / SPANS), session_id=session)],
    )


def answers(report: Mapping[str, Any]) -> str:
    """A markdown table of each cell's value and the pages that state it."""
    rows = [
        [cell, found["value"], ", ".join(found["sources"])] if found else [cell, "unsettled", ""]
        for cell, found in report.items()
    ]
    return table(["cell", "value", "sources"], rows)


@dataclass
class Tally:
    """One op's spans in a run: how many, how many the op cache answered, tokens and seconds."""

    asked: int = 0
    cached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0


def summary(spans: Sequence[Mapping[str, Any]]) -> str:
    """A markdown table of one run's spans by op, the last term of each span's placed key."""
    tallies: dict[str, Tally] = {}
    for span in spans:
        attributes = span.get("attributes", {})
        if (stored := attributes.get("effective.key")) is None:
            continue
        tally = tallies.setdefault(Key.parse(stored).terms()[-1].render(), Tally())
        tally.asked += 1
        tally.cached += bool(attributes.get("effective.reused"))
        tally.tokens_in += attributes.get("gen_ai.usage.input_tokens", 0)
        tally.tokens_out += attributes.get("gen_ai.usage.output_tokens", 0)
        tally.seconds += (span["endTimeUnixNano"] - span["startTimeUnixNano"]) / 1e9
    head = ["op", "asked", "from cache", "input tokens", "output tokens", "seconds"]
    rows = [
        [op, s.asked, s.cached, s.tokens_in, s.tokens_out, t"{s.seconds:.1f}"]
        for op, s in sorted(tallies.items())
    ]
    return table(head, rows)


def cells_of(pairs: list[str]) -> dict[str, str]:
    cells = {}
    for pair in pairs:
        name, sep, asks = pair.partition("=")
        if not sep or not name or not asks:
            raise SystemExit(f'a cell is name="what it asks", not {pair!r}')
        cells[name] = asks
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.deep_research")
    parser.add_argument("question")
    parser.add_argument("cells", nargs="+")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--dir", type=Path, default=Path(".deep_research"))
    parser.add_argument("--held", action="store_true")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    cells = cells_of(args.cells)
    args.dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    answer = live(args.dir, args.model, args.held, session=run_id)
    report = investigate(args.dir / "runs.db", args.question, cells, args.steps, answer, run_id)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(answers(report))
        if (spans := args.dir / SPANS).exists():
            print()
            print(summary(sidecar_spans(spans, session_id=run_id)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
