"""The web tools without the network: pages as HTML strings, and responses handed to `get`."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from effective.cache import Cache, FileStore
from effective.cost import Usage
from effective.domain import CallTool, DomainOp
from effective.interpreters import web
from effective.interpreters.web import (
    LIMIT,
    BraveSearch,
    Fetched,
    HeldSearch,
    NotHeld,
    Page,
    SearchRefused,
    Unreached,
    Unreadable,
    host,
    lasting,
    links_of,
    retrieve,
    text_of,
)
from effective.spend import TokenBudget, TokenBudgetExhausted

PAGE = """<html><head><title>skipped</title><style>p {}</style></head><body>
<nav><a href="/about">About</a></nav>
<main><p>Model X entered beta.</p><script>var x;</script>
<a href="https://changelog.example/v1#results">the changelog</a>
<a href="mailto:press@example.com">press</a>
<a href="/about">About again</a></main>
<footer>Model X entered beta.</footer></body></html>"""


def test_a_page_reads_as_its_text_without_scripts_or_styles():
    assert text_of(PAGE) == (
        "About Model X entered beta. the changelog press About again Model X entered beta."
    )


def test_the_main_text_leaves_out_layout_and_a_repeated_passage():
    assert text_of(PAGE, main=True) == "Model X entered beta. the changelog press About again"


def test_text_is_cut_at_the_limit():
    assert len(text_of("<p>" + "word " * LIMIT + "</p>")) == LIMIT


def test_links_are_absolute_web_links_kept_once_with_their_first_anchor_text():
    links = links_of(PAGE, "https://vendor.example/news/1")
    assert [(link.url, link.text) for link in links] == [
        ("https://vendor.example/about", "About"),
        ("https://changelog.example/v1", "the changelog"),
    ]


def answering(status: int, body: str, kind: str = "text/html"):
    def get(url: str) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(status, headers={"content-type": kind}, text=body, request=request)

    return get


def test_a_page_that_answers_is_its_text_and_links():
    url = "https://vendor.example/"
    assert retrieve(url, get=answering(200, PAGE)) == Fetched.of(
        url,
        Page(
            code=200,
            content_type="text/html",
            text=text_of(PAGE),
            final=url,
            links=links_of(PAGE, url),
        ),
    )


@pytest.mark.parametrize(
    ("get", "outcome"),
    [
        (answering(404, "gone"), Unreadable(code=404, content_type="text/html")),
        (
            answering(200, "%PDF", kind="application/pdf"),
            Unreadable(code=200, content_type="application/pdf"),
        ),
        (answering(503, "", kind=""), Unreadable(code=503, content_type="")),
    ],
    ids=["not found", "not html", "no content type"],
)
def test_a_page_that_does_not_answer_with_html_is_unreadable(get, outcome):
    url = "https://vendor.example/"
    assert retrieve(url, get=get) == Fetched.of(url, outcome)


def test_a_page_that_cannot_be_reached_says_why():
    def unreachable(url: str) -> httpx.Response:
        raise httpx.ConnectError("refused")

    url = "https://vendor.example/"
    assert retrieve(url, get=unreachable) == Fetched.of(url, Unreached(reason="ConnectError"))


def test_the_fetch_tool_asks_for_the_url_and_the_main_text(monkeypatch):
    asked: list[tuple[str, bool]] = []

    def recorded(url: str, main: bool = False) -> Fetched:
        asked.append((url, main))
        return Fetched.of(url, Unreached(reason="stubbed"))

    monkeypatch.setattr(web, "retrieve", recorded)
    web.fetch(
        CallTool(
            name="fetch", args={"url": "https://a.example/", "main": True}, result_schema=Fetched
        )
    )
    web.fetch(CallTool(name="fetch", args={"url": "https://b.example/"}, result_schema=Fetched))
    assert asked == [("https://a.example/", True), ("https://b.example/", False)]


@pytest.mark.parametrize(
    ("url", "name"),
    [("https://www.Example.com/x", "example.com"), ("example.com", "example.com"), ("", "")],
)
def test_a_host_is_lowercase_without_www(url, name):
    assert host(url) == name


RESULTS = {
    "web": {
        "results": [
            {"title": "Model X", "url": "https://vendor.example/x", "page_age": "2026-09-01T00"},
            {"title": "Release", "url": "https://changelog.example/v1", "age": "3 days ago"},
        ]
    }
}


class Brave:
    """Brave's API, answering every request with `RESULTS` or with `status`."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.sent: list[str] = []

    def __call__(self, url: str, key: str) -> httpx.Response:
        self.sent.append(url)
        headers = {"x-ratelimit-remaining": "999", "content-type": "application/json"}
        return httpx.Response(self.status, headers=headers, json=RESULTS)


def search(*queries: str) -> CallTool[Any]:
    return CallTool(name="search", args={"queries": list(queries)}, result_schema=web.Searched)


def test_a_query_is_paid_for_once_and_answered_from_disk_after(tmp_path: Path):
    brave = Brave()
    budget = TokenBudget(tmp_path / "requests", limit=5)
    tool = BraveSearch("secret-key", budget, tmp_path / "raw", get=brave)
    first, again = tool(search("asset x")), tool(search("asset x"))
    assert [hit.url for hit in first.hits] == [
        "https://vendor.example/x",
        "https://changelog.example/v1",
    ]
    assert [hit.age for hit in first.hits] == ["2026-09-01", "3 days ago"]
    assert (first.sent, first.kept) == (1, False)
    assert (again.sent, again.kept, again.hits) == (0, True, first.hits)
    assert len(brave.sent) == 1
    assert budget.spent == 1


def test_the_key_is_never_part_of_what_is_kept(tmp_path: Path):
    BraveSearch("secret-key", TokenBudget(tmp_path / "r", 5), tmp_path / "raw", get=Brave())(
        search("asset x")
    )
    (kept,) = (tmp_path / "raw").iterdir()
    assert "secret-key" not in kept.read_text()
    assert json.loads(kept.read_text())["ratelimit"] == {"x-ratelimit-remaining": "999"}


def test_a_refused_request_is_charged_and_raises(tmp_path: Path):
    budget = TokenBudget(tmp_path / "requests", limit=5)
    tool = BraveSearch("k", budget, tmp_path / "raw", get=Brave(status=429))
    with pytest.raises(SearchRefused, match="429"):
        tool(search("asset x"))
    assert budget.spent == 1
    assert not (tmp_path / "raw").exists()


def test_a_spent_budget_refuses_before_sending(tmp_path: Path):
    brave = Brave()
    tool = BraveSearch(
        "k", TokenBudget(tmp_path / "requests", limit=0), tmp_path / "raw", get=brave
    )
    with pytest.raises(TokenBudgetExhausted):
        tool(search("asset x"))
    assert brave.sent == []


def test_a_request_lost_in_transit_is_still_charged(tmp_path: Path):
    def lost(url: str, key: str) -> httpx.Response:
        raise httpx.ReadTimeout("lost")

    budget = TokenBudget(tmp_path / "requests", limit=5)
    with pytest.raises(httpx.ReadTimeout):
        BraveSearch("k", budget, tmp_path / "raw", get=lost)(search("asset x"))
    assert budget.spent == 1


def test_a_response_that_does_not_read_as_hits_is_not_kept(tmp_path: Path):
    def untitled(url: str, key: str) -> httpx.Response:
        return httpx.Response(200, json={"web": {"results": [{"url": "https://a.example/"}]}})

    tool = BraveSearch("k", TokenBudget(tmp_path / "r", 5), tmp_path / "raw", get=untitled)
    with pytest.raises(KeyError):
        tool(search("asset x"))
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("count", [0, 21])
def test_brave_answers_one_to_twenty_results_a_request(tmp_path: Path, count):
    with pytest.raises(ValueError, match="1 to 20"):
        BraveSearch("k", TokenBudget(tmp_path / "r", 5), tmp_path / "raw", count=count)


def test_a_held_search_replays_what_was_kept_without_a_key(tmp_path: Path):
    paid = BraveSearch("k", TokenBudget(tmp_path / "r", 5), tmp_path / "raw", get=Brave())
    answered = paid(search("asset x"))
    held = HeldSearch(tmp_path / "raw")(search("asset x"))
    assert (held.hits, held.kept, held.sent) == (answered.hits, True, 0)


def test_a_held_search_refuses_a_query_nobody_kept(tmp_path: Path):
    with pytest.raises(NotHeld, match="asset y"):
        HeldSearch(tmp_path / "raw")(search("asset y"))


LASTING_PAGE = Page(code=200, content_type="text/html", text="t", final="https://a.example/")


@pytest.mark.parametrize(
    ("outcome", "transient"),
    [
        (LASTING_PAGE, False),
        (Unreadable(code=403, content_type="text/html"), False),
        (Unreadable(code=404, content_type="text/html"), False),
        (Unreadable(code=429, content_type=""), True),
        (Unreadable(code=503, content_type="text/html"), True),
        (Unreadable(code=520, content_type="text/html"), True),
        (Unreadable(code=507, content_type=""), True),
        (Unreached(reason="ConnectTimeout"), True),
    ],
    ids=[
        "a page",
        "forbidden",
        "gone",
        "rate limited",
        "unavailable",
        "an origin error",
        "insufficient storage",
        "unreached",
    ],
)
def test_a_fetch_is_transient_when_fetching_again_may_differ(outcome, transient):
    assert Fetched.of("https://a.example/", outcome).transient is transient
    assert lasting(fetching("https://a.example/"), Fetched.of("https://a.example/", outcome)) is (
        not transient
    )


def test_an_answer_that_is_not_a_fetch_lasts():
    assert lasting(search("q"), web.Searched(hits=[]))


def fetching(url: str) -> CallTool[Any]:
    return CallTool(name="fetch", args={"url": url}, result_schema=Fetched)


class Site:
    """A metered domain answering every fetch with `outcome`, and counting the asks."""

    def __init__(self, outcome: Page | Unreadable | Unreached) -> None:
        self.outcome = outcome
        self.asked = 0

    def run_metered(self, op: DomainOp[Any]) -> tuple[Fetched, Usage]:
        match op:
            case CallTool(name="fetch", args={"url": str(url)}):
                self.asked += 1
                return Fetched.of(url, self.outcome), Usage()
        raise TypeError(f"the site answers fetch, not {op!r}")


@pytest.mark.parametrize(
    ("outcome", "asked"),
    [(LASTING_PAGE, 1), (Unreadable(code=503, content_type="text/html"), 2)],
    ids=["a page is served from the store", "a 503 is asked again"],
)
def test_the_op_cache_keeps_a_lasting_fetch_and_not_a_transient_one(tmp_path, outcome, asked):
    site = Site(outcome)
    cached = Cache(FileStore(tmp_path), {"fetch": "httpx"}, keeps=lasting).over(site)
    for _ in range(2):
        cached.run_metered(fetching("https://a.example/"))
    assert site.asked == asked
