"""The policy comparison against a corpus crawled from the Kestrel world.

Predicted before the rows ran: the crawl explores every lead the policies can reach in three
steps, so each policy replays with no miss and settles both cells; an empty corpus answers every
op as a miss and settles nothing; and a replay leaves the corpus as it found it.
"""

import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from test_deep_research import CELLS, QUESTION, World

from effective.cache import Cache, FileStore
from effective.cost import MeteredInterpreter, Usage
from effective.domain import AskLLM, CallTool, Judge
from effective.interpreters.jev import MODEL as JEV_MODEL
from effective.interpreters.web import FETCH_TOOL, BraveSearch
from effective.search import everything
from effective.spend import TokenBudget
from examples.deep_research import __main__ as command
from examples.deep_research.compare import POLICIES, READER, Scored, main, replay, table

TRUTH = {"launch": "2031", "operator": "orbital survey agency"}


def crawled(root: Path, steps: int = 3, reader: str = READER, judge: str = JEV_MODEL) -> Path:
    """The Kestrel world crawled breadth first for `steps` into a corpus at `root`, its searches
    kept through `BraveSearch` as a live crawl keeps them."""
    world = World()

    def brave(url: str, key: str) -> httpx.Response:
        (query,) = parse_qs(urlsplit(url).query)["q"]
        results = [{"title": page, "url": page} for page in world.urls(query)]
        return httpx.Response(200, json={"web": {"results": results}})

    search = BraveSearch(
        "key", TokenBudget(root / "brave-requests", 100), root / "search", get=brave
    )

    def tool(op: CallTool[Any]) -> Any:
        return search(op) if op.name == "search" else world.run(op)

    answered: dict[type | str, str] = {AskLLM: reader, Judge: judge, FETCH_TOOL: "httpx"}
    answer = MeteredInterpreter(
        llm=lambda op: (world.run(op), Usage()),
        tools=tool,
        judge=lambda op: (world.run(op), Usage()),
        cache=Cache(FileStore(root / "ops"), answered),
    )
    command.investigate(
        root / "crawl.db", QUESTION, CELLS, steps, answer, pick=everything, until_settled=False
    )
    return root


def test_every_policy_replays_the_crawl_without_a_miss(tmp_path):
    scored = replay(crawled(tmp_path), QUESTION, CELLS, 3, TRUTH)
    assert [s.policy for s in scored] == list(POLICIES)
    assert {s.policy: (s.correct, s.misses) for s in scored} == {
        policy: (2, {}) for policy in POLICIES
    }


def test_an_empty_corpus_misses_every_op_and_settles_nothing(tmp_path):
    scored = replay(tmp_path, QUESTION, CELLS, 3, TRUTH)
    assert all(s.correct == 0 and s.settled == 0 for s in scored)
    assert all(s.misses["search"] == s.asked["search"] > 0 for s in scored)


def test_a_replay_that_misses_writes_nothing_into_the_corpus(tmp_path):
    """One step crawled, three replayed: the misses' empty answers must not become the corpus."""
    corpus = crawled(tmp_path, steps=1)
    before = sorted(p.relative_to(corpus) for p in (corpus / "ops").rglob("*"))
    scored = replay(corpus, QUESTION, CELLS, 3, TRUTH)
    assert all(sum(s.misses.values()) > 0 for s in scored)
    assert sorted(p.relative_to(corpus) for p in (corpus / "ops").rglob("*")) == before


def cells(line: str) -> list[str]:
    """A markdown table row's cells, split at unescaped pipes and stripped."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


def test_the_table_shows_each_op_asked_and_missed(tmp_path):
    scored = replay(crawled(tmp_path), QUESTION, CELLS, 3, TRUTH)
    head, rule, first, *_ = table(scored, 2).splitlines()
    assert cells(head) == [
        "policy",
        "correct",
        "steps",
        "searches",
        "fetches",
        "reads",
        "judgments",
    ]
    assert set("".join(cells(rule))) == {"-"}
    assert cells(first)[:2] == ["worth, corroborating", "2 of 2"]
    assert all(cell.endswith("(0 missed)") for cell in cells(first)[3:])


def test_a_pipe_in_a_policy_name_stays_inside_its_cell():
    scored = [
        Scored("a | b", 1, 1, 2, {k: 0 for k in ("search", "fetch", "read", "judgment")}, {})
    ]
    _, _, row = table(scored, 2).splitlines()
    assert cells(row)[0] == "a \\| b"
    assert len(cells(row)) == 7


@pytest.mark.parametrize(
    ("crawl", "missed"),
    [
        ({"reader": "another model"}, "read"),
        ({"judge": "another judge"}, "judgment"),
    ],
)
def test_an_op_answered_by_another_model_is_a_miss(tmp_path, crawl, missed):
    """The corpus names each answer by what answered it, so a replay asking a different model
    finds none of that model's answers there."""
    (worth, *_) = replay(crawled(tmp_path, **crawl), QUESTION, CELLS, 3, TRUTH)
    assert worth.misses[missed] == worth.asked[missed] > 0
    assert worth.misses.get("fetch", 0) == 0


def test_a_corpus_without_its_op_cache_misses_every_fetch(tmp_path):
    corpus = crawled(tmp_path)
    shutil.rmtree(corpus / "ops")
    (worth, *_) = replay(corpus, QUESTION, CELLS, 3, TRUTH)
    assert worth.misses["fetch"] == worth.asked["fetch"] > 0
    assert worth.misses.get("search", 0) == 0


def test_the_replay_command_prints_the_comparison(tmp_path, capsys):
    corpus = crawled(tmp_path)
    argv = ["replay", QUESTION, *(f"{n}={a}" for n, a in CELLS.items()), "--dir", str(corpus)]
    argv += [f"--truth={cell}={value}" for cell, value in TRUTH.items()]
    assert main(argv) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [cells(line)[0] for line in lines[2:]] == list(POLICIES)
