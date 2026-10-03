"""Compare research policies on a frozen corpus.

    python -m examples.deep_research.compare crawl "PEP 750 template strings" \\
        version="..." accepted="..." --dir CORPUS [--steps 3] [--searches 60]
    python -m examples.deep_research.compare replay "PEP 750 template strings" \\
        version="..." accepted="..." --dir CORPUS --truth version=3.14 --truth accepted=2025

`crawl` is live and spends: it explores breadth first, every queued lead at every step, past the
point where cells settle, so every search, page, reading and judgment it reaches is kept in the
corpus, and it prints a summary of the ops it asked from the spans it writes beside the corpus.
`replay` spends nothing: each policy runs against the corpus as a closed world, where an op the
corpus holds is answered from it and any other op is answered empty and counted as a miss. A
policy's misses say how far it left the ground the crawl covered.

The corpus is one crawl of one question, so a replay is deterministic: it compares policies on
that corpus, and says nothing of how a policy varies with a live reader.
"""

import argparse
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from effective.cache import Cache, FileStore
from effective.cost import MeteredInterpreter, Usage
from effective.domain import Answers, AskLLM, CallTool, Judge
from effective.interpreters.jev import MODEL as JEV_MODEL
from effective.interpreters.web import (
    FETCH_TOOL,
    SEARCH_TOOL,
    Fetched,
    HeldSearch,
    NotHeld,
    Searched,
    Unreached,
)
from effective.markdown import table as markdown_table
from effective.search import everything
from effective.telemetry import sidecar_spans
from examples.deep_research.__main__ import SPANS, cells_of, investigate, live, summary
from examples.deep_research.research import UNSCORED, Read, in_order, worth_most

READER = "haiku"
"""The model the corpus's readings were asked of, which names them in the op cache."""

POLICIES: dict[str, dict[str, Any]] = {
    "worth, corroborating": {"pick": worth_most, "corroborate": True},
    "worth": {"pick": worth_most, "corroborate": False},
    "in order, corroborating": {"pick": in_order, "corroborate": True},
    "breadth, corroborating": {"pick": everything, "corroborate": True},
}


@dataclass(frozen=True)
class ReadOnly:
    """A store that answers from `inner` and keeps nothing, so a replay never writes the corpus."""

    inner: FileStore

    def read(self, digest: str) -> bytes | None:
        return self.inner.read(digest)

    def write(self, digest: str, value: bytes) -> None:
        return None


@dataclass
class Closed:
    """The corpus at `root` as a closed world, counting each op it does not hold by kind."""

    root: Path
    misses: Counter[str] = field(default_factory=Counter)

    def interpreter(self) -> MeteredInterpreter:
        held = HeldSearch(self.root / "search")

        def tool(op: CallTool[Any]) -> Any:
            match op.name:
                case name if name == SEARCH_TOOL:
                    try:
                        return held(op)
                    except NotHeld:
                        self.misses["search"] += 1
                        return Searched(hits=[])
                case name if name == FETCH_TOOL:
                    self.misses["fetch"] += 1
                    return Fetched.of(op.args["url"], Unreached(reason="not in the corpus"))
            raise LookupError(f"the corpus answers search and fetch, not {op.name!r}")

        def read(op: AskLLM[Any]) -> tuple[Read, Usage]:
            self.misses["read"] += 1
            return Read(claims=[], leads=[]), Usage()

        def judge(op: Judge[Any]) -> tuple[Answers, Usage]:
            self.misses["judgment"] += 1
            return Answers.model_validate({"settles": {"p": UNSCORED}}), Usage()

        answered: dict[type | str, str] = {AskLLM: READER, Judge: JEV_MODEL, FETCH_TOOL: "httpx"}
        store = ReadOnly(FileStore(self.root / "ops"))
        return MeteredInterpreter(llm=read, tools=tool, judge=judge, cache=Cache(store, answered))


@dataclass(frozen=True)
class Scored:
    policy: str
    correct: int
    settled: int
    steps: int
    asked: Mapping[str, int]
    misses: Mapping[str, int]


ASKED = {"search": "tool:search", "fetch": "tool:fetch", "read": "step:read", "judgment": "judge:"}


def replay(
    root: Path, question: str, cells: Mapping[str, str], steps: int, truth: Mapping[str, str]
) -> list[Scored]:
    """Each policy against the corpus at `root`, as its own durable task in a fresh store."""
    scored = []
    for name, policy in POLICIES.items():
        closed = Closed(root)
        with tempfile.TemporaryDirectory() as scratch:
            store = Path(scratch) / "runs.db"
            report = investigate(store, question, cells, steps, closed.interpreter(), **policy)
            with closing(sqlite3.connect(store)) as db:
                names = [row[0] for row in db.execute("SELECT name FROM checkpoints")]
        answers = {cell: (found or {}).get("value") for cell, found in report.items()}
        scored.append(
            Scored(
                policy=name,
                correct=sum(answers.get(cell) == value for cell, value in truth.items()),
                settled=sum(value is not None for value in answers.values()),
                steps=len({n.split(";")[0] for n in names if n.startswith("d:")}),
                asked={kind: sum(mark in n for n in names) for kind, mark in ASKED.items()},
                misses=dict(closed.misses),
            )
        )
    return scored


HEADS = {"search": "searches", "fetch": "fetches", "read": "reads", "judgment": "judgments"}


def table(scored: list[Scored], cells: int) -> str:
    """A markdown table: each policy's correct cells and steps, and each kind of op it asked, with
    how many of those the corpus did not hold."""
    head = ["policy", "correct", "steps", *HEADS.values()]
    rows = [
        [
            s.policy,
            t"{s.correct} of {cells}",
            s.steps,
            *(t"{s.asked[kind]} ({s.misses.get(kind, 0)} missed)" for kind in HEADS),
        ]
        for s in scored
    ]
    return markdown_table(head, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.deep_research.compare")
    parser.add_argument("mode", choices=("crawl", "replay"))
    parser.add_argument("question")
    parser.add_argument("cells", nargs="+")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--searches", type=int, default=60)
    parser.add_argument("--truth", action="append", default=[])
    args = parser.parse_args(argv)
    cells = cells_of(args.cells)
    args.dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "crawl":
        run_id = str(uuid4())
        answer = live(args.dir, READER, held=False, searches=args.searches, session=run_id)
        investigate(
            args.dir / "crawl.db",
            args.question,
            cells,
            args.steps,
            answer,
            run_id,
            pick=everything,
            until_settled=False,
        )
        print(summary(sidecar_spans(args.dir / SPANS, session_id=run_id)))
        return 0
    truth = cells_of(args.truth)
    print(table(replay(args.dir, args.question, cells, args.steps, truth), len(truth)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
