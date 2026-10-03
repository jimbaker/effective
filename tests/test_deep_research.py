"""The deep-research example against a fictional world: the Kestrel probe, on `.example` hosts.

Predicted from `PAGES` and `READS` before the rows ran:

| the claim                                         | the number or answer                       |
|---------------------------------------------------|--------------------------------------------|
| the first step leaves `launch` contested          | 2031 on news, 2029 on blog                 |
| the first step settles `operator`                 | agency and news agree                      |
| the second step settles `launch`                  | wiki is the second host for 2031           |
| the run stops when every cell is settled          | after two steps, with leads still queued   |
| a claim whose quote is not on its page            | never reaches the evidence                 |
| one step only                                     | `launch` unsettled, `operator` settled     |
| one query suggested for two cells in one step     | two leads, and the run completes           |
| both cells quoted only by news after one step     | corroborating searches away from news next |
"""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from _shapes import Shape, run, sweep

from effective.api import Effect
from effective.cache import Cache, FileStore
from effective.cost import MeteredInterpreter, Usage
from effective.domain import Answers, AskLLM, CallTool, DomainOp, Judge
from effective.interpreters.web import Fetched, Hit, NotHeld, Page, Searched, brave_url
from effective.search import Round
from effective.telemetry import otlp_jsonl_sink, sidecar_spans, traced
from examples.deep_research import __main__ as command
from examples.deep_research.research import (
    Cited,
    Claim,
    Evidence,
    Lead,
    Next,
    Read,
    alone,
    contested,
    into_evidence,
    read,
    research,
    settled_cells,
)

QUESTION = "Kestrel probe"
CELLS = {
    "launch": "the year the Kestrel probe launched",
    "operator": "the agency that operates the Kestrel probe",
}
NEWS = "https://news.example/kestrel"
AGENCY = "https://agency.example/missions/kestrel"
BLOG = "https://blog.example/kestrel-myths"
WIKI = "https://wiki.example/Kestrel"

PAGES = {
    NEWS: "The Kestrel probe launched in 2031. It is run by the Orbital Survey Agency.",
    AGENCY: "Kestrel is operated by the Orbital Survey Agency from its northern station.",
    BLOG: "A popular myth says Kestrel launched in 2029, years before its real flight.",
    WIKI: "Kestrel is a survey probe. Launched: 2031. Operator: Orbital Survey Agency.",
}

LAUNCH_SEED = "Kestrel probe the year the Kestrel probe launched"
OPERATOR_SEED = "Kestrel probe the agency that operates the Kestrel probe"
SUGGESTED = "Kestrel launch date encyclopedia"

SEARCHES = {
    LAUNCH_SEED: [BLOG, NEWS],
    OPERATOR_SEED: [AGENCY, NEWS],
    SUGGESTED: [WIKI],
}


def refutes(query: str) -> bool:
    """A search that names both launch years, as the refuting lead for a contested cell does."""
    return "2029" in query and "2031" in query


READS = {
    NEWS: Read(
        claims=[
            Claim(cell="launch", value="2031", quote="launched in 2031"),
            Claim(
                cell="operator", value="Orbital Survey Agency", quote="the Orbital Survey Agency"
            ),
        ],
        leads=[Next(cell="launch", query=SUGGESTED)],
    ),
    AGENCY: Read(
        claims=[
            Claim(
                cell="operator",
                value="Orbital Survey Agency",
                quote="operated by the Orbital Survey Agency",
            )
        ],
        leads=[Next(cell="operator", query="Orbital Survey Agency missions")],
    ),
    BLOG: Read(
        claims=[
            Claim(cell="launch", value="2029", quote="Kestrel launched in 2029"),
            Claim(
                cell="operator", value="Deep Space Office", quote="run by the Deep Space Office"
            ),
        ],
        leads=[Next(cell="operator", query=SUGGESTED)],
    ),
    WIKI: Read(claims=[Claim(cell="launch", value="2031", quote="Launched: 2031")], leads=[]),
}


def html_page(url: str, text: str) -> Fetched:
    """A page that answered 200 with HTML whose text is `text`."""
    return Fetched.of(url, Page(code=200, content_type="text/html", text=text, final=url))


class World:
    """Search, fetch, a reader and Jev, each answering from the tables above."""

    def __init__(self) -> None:
        self.searched: list[str] = []
        self.read: list[str] = []

    def urls(self, query: str) -> list[str]:
        """The pages a search for `query` finds."""
        return SEARCHES.get(query) or ([WIKI] if refutes(query) else [])

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case CallTool(name="search", args={"queries": [str(query)]}):
                self.searched.append(query)
                hits = [Hit(query=query, title=url, url=url) for url in self.urls(query)]
                return Searched(hits=hits)
            case CallTool(name="fetch", args={"url": str(url)}):
                return html_page(url, PAGES[url])
            case AskLLM(messages=messages):
                content = " ".join(message.content for message in messages)
                (url,) = [url for url in PAGES if url in content]
                self.read.append(url)
                return READS[url]
            case Judge():
                return Answers.model_validate({"settles": {"p": 0.6}})
        raise TypeError(f"the Kestrel world does not answer {op!r}")


ANSWER = {
    "launch": {"value": "2031", "sources": [NEWS, WIKI]},
    "operator": {"value": "orbital survey agency", "sources": [AGENCY, NEWS]},
}


def investigate(_run_id: str) -> Effect[dict[str, Any]]:
    return (yield from research(QUESTION, CELLS))


RESEARCH = Shape(spellings={"frontier": investigate}, domain=World, answer=lambda: ANSWER)


def test_the_research_settles_every_cell_with_two_hosts(backend):
    outcome = run(backend, investigate, World())
    assert outcome.snap.result == ANSWER


def test_a_contested_cell_sends_a_refuting_search(backend):
    world = World()
    run(backend, investigate, world)
    assert world.searched[:2] == [LAUNCH_SEED, OPERATOR_SEED]
    assert len(world.searched) == 4
    assert SUGGESTED in world.searched[2:]
    assert any(refutes(query) for query in world.searched[2:])


def test_one_step_leaves_the_contested_cell_unsettled(backend):
    def hurried(_run_id: str) -> Effect[dict[str, Any]]:
        return (yield from research(QUESTION, CELLS, steps=1))

    outcome = run(backend, hurried, World())
    assert outcome.snap.result == {"launch": None, "operator": ANSWER["operator"]}


def test_a_claim_quoting_words_not_on_its_page_is_dropped(backend):
    def blog(_run_id: str) -> Effect[list[Any]]:
        page = html_page(BLOG, PAGES[BLOG])
        finding = yield from read(QUESTION, CELLS, Lead("launch", LAUNCH_SEED), page)
        return [[claim.cell, claim.value] for claim in finding.claims]

    outcome = run(backend, blog, World())
    assert outcome.snap.result == [["launch", "2029"]]


def test_a_crash_after_every_step_converges(backend):
    sweep(backend, RESEARCH, "frontier")


def away_from_news(query: str, names: str) -> bool:
    """A search that leaves out news.example and names `names`."""
    return "-site:news.example" in query and names in query.lower()


class Lonely(World):
    """Both seeds find only the news page, and only a search away from it finds another host."""

    def urls(self, query: str) -> list[str]:
        if query in (LAUNCH_SEED, OPERATOR_SEED):
            return [NEWS]
        if away_from_news(query, "2031"):
            return [WIKI]
        if away_from_news(query, "orbital survey agency"):
            return [AGENCY]
        return []


def test_a_cell_one_host_quotes_sends_a_search_away_from_that_host(backend):
    world = Lonely()
    outcome = run(backend, investigate, world)
    assert world.searched[:2] == [LAUNCH_SEED, OPERATOR_SEED]
    launch, operator = world.searched[2:]
    assert away_from_news(launch, "2031")
    assert away_from_news(operator, "orbital survey agency")
    assert outcome.snap.result == {
        "launch": {"value": "2031", "sources": [NEWS, WIKI]},
        "operator": {"value": "orbital survey agency", "sources": [AGENCY, NEWS]},
    }


class Reader(World):
    """A reader that reports one claim, whatever the page."""

    def __init__(self, claim: Claim) -> None:
        super().__init__()
        self.claim = claim

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM():
                return Read(claims=[self.claim], leads=[])
        return super().run(op)


@pytest.mark.parametrize(
    ("quote", "kept"),
    [
        ("Launched: 2031", True),
        ("Kestrel is a survey probe", False),
        ("Kestrel is a survey probe. ... Launched: 2031", True),
        ("Launched: 2031 ... Kestrel is a survey probe", False),
    ],
    ids=["states the value", "names no value", "elided, in order", "elided, out of order"],
)
def test_a_claim_is_kept_when_its_quote_states_the_value_on_the_page(backend, quote, kept):
    claim = Claim(cell="launch", value="2031", quote=quote)

    def wiki(_run_id: str) -> Effect[int]:
        page = html_page(WIKI, PAGES[WIKI])
        finding = yield from read(QUESTION, CELLS, Lead("launch", LAUNCH_SEED), page)
        return len(finding.claims)

    outcome = run(backend, wiki, Reader(claim))
    assert outcome.snap.result == int(kept)


def test_a_quote_matches_a_page_whose_markup_left_a_space_before_punctuation(backend):
    """Extraction across a code element reads `3.14 , which`; the reader quotes `3.14, which`."""
    claim = Claim(cell="launch", value="2031", quote="It launched in 2031, on time.")

    def split(_run_id: str) -> Effect[int]:
        page = html_page(WIKI, "It launched in 2031 , on time .")
        finding = yield from read(QUESTION, CELLS, Lead("launch", LAUNCH_SEED), page)
        return len(finding.claims)

    assert run(backend, split, Reader(claim)).snap.result == 1


def evidence(*cited: tuple[str, str, str]) -> Evidence:
    return Evidence(QUESTION, CELLS, cited=tuple(Cited(*claim) for claim in cited))


def test_a_value_two_hosts_quote_settles_its_cell_against_one_rival():
    found = evidence(("launch", "2031", NEWS), ("launch", "2031", WIKI), ("launch", "2029", BLOG))
    assert settled_cells(found) == {"launch"}
    assert contested(found) == {}


def test_two_values_with_equal_support_leave_the_cell_contested():
    found = evidence(
        ("launch", "2031", NEWS),
        ("launch", "2031", WIKI),
        ("launch", "2029", BLOG),
        ("launch", "2029", AGENCY),
    )
    assert settled_cells(found) == set()
    assert contested(found) == {"launch": ["2029", "2031"]}


def test_a_cell_is_alone_with_one_value_on_too_few_hosts():
    one = evidence(("launch", "2031", NEWS), ("operator", "Orbital Survey Agency", NEWS))
    two = evidence(("launch", "2031", NEWS), ("launch", "2031", WIKI))
    rival = evidence(("launch", "2031", NEWS), ("launch", "2029", BLOG))
    assert alone(one) == {
        "launch": ("2031", {"news.example"}),
        "operator": ("orbital survey agency", {"news.example"}),
    }
    assert alone(two) == {}
    assert alone(rival) == {}


def test_one_host_quoting_a_value_twice_is_one_host():
    found = evidence(("launch", "2031", NEWS), ("launch", "2031.", NEWS))
    assert settled_cells(found) == set()


def answering(world: World) -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda op: (world.run(op), Usage()),
        tools=world.run,
        judge=lambda op: (world.run(op), Usage()),
    )


def test_the_command_runs_the_research_as_one_durable_task(tmp_path):
    report = command.investigate(tmp_path / "runs.db", QUESTION, CELLS, 6, answering(World()))
    assert report == ANSWER


class Offline(World):
    def run(self, op: DomainOp[Any]) -> Any:
        raise ConnectionError("the search is down")


def test_a_research_that_fails_says_so(tmp_path):
    with pytest.raises(RuntimeError, match="did not complete"):
        command.investigate(tmp_path / "runs.db", QUESTION, CELLS, 6, answering(Offline()))


def test_the_command_prints_the_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(command, "live", lambda root, model, held, **_: answering(World()))
    argv = [QUESTION, *(f"{name}={asks}" for name, asks in CELLS.items()), "--dir", str(tmp_path)]
    argv.append("--json")
    assert command.main(argv) == 0
    assert json.loads(capsys.readouterr().out) == ANSWER


def test_a_cell_names_what_it_asks():
    assert command.cells_of(["launch=the year it launched", "operator=who runs it"]) == {
        "launch": "the year it launched",
        "operator": "who runs it",
    }


@pytest.mark.parametrize("pair", ["launch", "=the year", "launch="])
def test_a_cell_without_a_name_or_a_question_is_refused(pair):
    with pytest.raises(SystemExit, match="name="):
        command.cells_of([pair])


@pytest.mark.parametrize(
    ("held", "missing"), [(False, r"BRAVE\w*_API_KEY"), (True, "JEV_API_KEY")]
)
def test_a_live_run_without_its_keys_stops_before_it_starts(monkeypatch, tmp_path, held, missing):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(SystemExit, match=missing):
        command.live(tmp_path, "haiku", held)


def test_a_held_run_refuses_a_search_nobody_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_API_KEY", "jev")
    answer = command.live(tmp_path, "haiku", held=True)
    with pytest.raises(NotHeld):
        answer.run(CallTool(name="search", args={"queries": ["q"]}, result_schema=Searched))


def test_a_live_run_serves_a_kept_search_without_sending(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAVE_API_KEY", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave")
    monkeypatch.setenv("JEV_API_KEY", "jev")
    kept = {"web": {"results": [{"title": "Kestrel", "url": NEWS}]}}
    url = brave_url("q", 10)
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / (hashlib.sha256(url.encode()).hexdigest() + ".json")).write_text(
        json.dumps(kept)
    )
    answer = command.live(tmp_path, "haiku", held=False)
    found = answer.run(CallTool(name="search", args={"queries": ["q"]}, result_schema=Searched))
    assert ([hit.url for hit in found.hits], found.kept, found.sent) == ([NEWS], True, 0)


def outside_quotes(query: str) -> str:
    """The query with its quoted phrases removed: what the search engine reads as operators."""
    return re.sub(r'"[^"]*"', "", query)


def test_a_value_read_off_a_page_cannot_forge_a_search_operator():
    """A reader's value reads `2031 -site:wiki.example`: the corroborating search quotes it as one
    phrase, so the only host it leaves out is the one that quoted the value."""
    forged = evidence(("launch", "2031 -site:wiki.example", NEWS))
    step = into_evidence(Round(value=forged, rest=[], expanded=[], scored=[]))
    (lead,) = step.queue
    assert '"2031 -site:wiki.example"' in lead.query
    assert outside_quotes(lead.query).split()[-1] == "-site:news.example"
    assert "wiki.example" not in outside_quotes(lead.query)


def test_a_lead_is_keyed_by_its_cell_and_the_digest_of_its_query():
    query = "pep 750"
    launch, operator = Lead("launch", query).key.stored(), Lead("operator", query).key.stored()
    assert launch.startswith("lead:launch,sha256-")
    assert operator.startswith("lead:operator,sha256-")
    assert launch.split(",")[1] == operator.split(",")[1]


def traced_run(root: Path, run_id: str) -> list[dict[str, Any]]:
    """The Kestrel world researched once as `run_id`, through an op cache and a span sidecar at
    `root`, and that run's span rows."""
    world = World()
    answered: dict[type | str, str] = {AskLLM: "reader", Judge: "jev", "fetch": "httpx"}
    answer = MeteredInterpreter(
        llm=lambda op: (world.run(op), Usage(prompt_tokens=100, completion_tokens=10)),
        tools=world.run,
        judge=lambda op: (world.run(op), Usage(prompt_tokens=30)),
        cache=Cache(FileStore(root / "ops"), answered),
        domain_layers=[traced(otlp_jsonl_sink(root / "spans.jsonl"), session_id=run_id)],
    )
    command.investigate(root / "runs.db", QUESTION, CELLS, 6, answer, run_id)
    return sidecar_spans(root / "spans.jsonl", session_id=run_id)


def test_the_summary_counts_each_op_and_what_the_cache_answered(tmp_path):
    traced_run(tmp_path, "first")
    second = traced_run(tmp_path, "second")
    rows = {cells(line)[0]: cells(line)[1:] for line in command.summary(second).splitlines()[2:]}
    assert set(rows) == {"judge:worth", "step:read", "tool:fetch", "tool:search"}
    for op in ("judge:worth", "step:read", "tool:fetch"):
        asked, cached, *_ = rows[op]
        assert asked == cached != "0"
    assert rows["tool:search"][1] == "0"


def test_every_span_joins_a_checkpoint_of_its_run(tmp_path):
    spans = traced_run(tmp_path, "joined")
    keys = {span["attributes"]["effective.key"] for span in spans}
    with closing(sqlite3.connect(tmp_path / "runs.db")) as db:
        (task,) = db.execute("SELECT task_id FROM tasks").fetchone()
        names = {
            row[0] for row in db.execute("SELECT name FROM checkpoints WHERE task_id=?", (task,))
        }
    assert keys
    assert keys <= names


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


def test_the_command_prints_its_answers_and_its_summary_as_markdown(monkeypatch, tmp_path, capsys):
    def traced_live(root: Path, model: str, held: bool, session: str, **_: Any):
        world = World()
        return MeteredInterpreter(
            llm=lambda op: (world.run(op), Usage()),
            tools=world.run,
            judge=lambda op: (world.run(op), Usage()),
            domain_layers=[traced(otlp_jsonl_sink(root / "spans.jsonl"), session_id=session)],
        )

    monkeypatch.setattr(command, "live", traced_live)
    argv = [QUESTION, *(f"{name}={asks}" for name, asks in CELLS.items()), "--dir", str(tmp_path)]
    assert command.main(argv) == 0
    answers, summary = capsys.readouterr().out.strip().split("\n\n")
    rows = [cells(line) for line in answers.splitlines()[2:]]
    assert rows == [
        ["launch", "2031", ", ".join(ANSWER["launch"]["sources"])],
        ["operator", "orbital survey agency", ", ".join(ANSWER["operator"]["sources"])],
    ]
    assert cells(summary.splitlines()[0])[:3] == ["op", "asked", "from cache"]
    assert {cells(line)[0] for line in summary.splitlines()[2:]} == {
        "judge:worth",
        "step:read",
        "tool:fetch",
        "tool:search",
    }
