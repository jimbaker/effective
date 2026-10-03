"""The web as two tools: `fetch` answers one page as text, and `search` answers web search results.

| tool     | answered by     | spends                                           |
|----------|-----------------|--------------------------------------------------|
| `fetch`  | `fetch`         | one request per page                             |
| `search` | `BraveSearch`   | one request per query, against a request budget  |
| `search` | `HeldSearch`    | nothing: only searches already kept on disk      |

A fetch comes to a `Page`, an `Unreadable` answer or no answer at all (`Unreached`), and each is an
answer: the workflow decides what an unreadable or unreached page means. A search response is kept
on disk under its request URL, so a query is paid for once, and `HeldSearch` replays a run's
searches without a key.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, Literal, assert_never
from urllib.parse import urlencode, urljoin, urlsplit

import httpx
from pydantic import BaseModel
from pydantic import Field as PydanticField

from effective.domain import CallTool, DomainOp
from effective.spend import TokenBudget

FETCH_TOOL = "fetch"
SEARCH_TOOL = "search"
AGENT = "effective-web/0.1"
LIMIT = 4000
"""Characters of text kept per page, which bounds the state a judgment reads."""
LINKS = 80
"""Links kept per page, in the page's order."""
_SKIP = {"script", "style", "noscript", "svg", "head", "template"}
_LAYOUT = {"nav", "header", "footer", "aside"}
"""Tags whose text is a site's layout rather than a page's content; their links still count."""


class Link(BaseModel):
    url: str
    text: str


class Page(BaseModel):
    """The server answered 200 with HTML: the page's text and links, as `text_of` and `links_of`
    read them, and the address it ended at after redirects."""

    kind: Literal["page"] = "page"
    code: int
    content_type: str
    text: str
    final: str
    links: list[Link] = []


class Unreadable(BaseModel):
    """The server answered, with a status other than 200 or with something other than HTML."""

    kind: Literal["unreadable"] = "unreadable"
    code: int
    content_type: str


class Unreached(BaseModel):
    """No answer: the request failed before a response, or nothing held the page."""

    kind: Literal["unreached"] = "unreached"
    reason: str


type Outcome = Annotated[Page | Unreadable | Unreached, PydanticField(discriminator="kind")]


TRANSIENT = frozenset({408, 425, 429})
"""HTTP statuses below 500 that a later request may not meet: a timeout, too early, a rate
limit. Every server error, 500 and up, is transient too."""


class Fetched(BaseModel):
    """What fetching `url` came to."""

    url: str
    outcome: Outcome

    @classmethod
    def of(cls, url: str, outcome: Page | Unreadable | Unreached) -> Fetched:
        return cls(url=url, outcome=outcome)

    @property
    def transient(self) -> bool:
        """Whether fetching again may come to something else: no answer, or a status in
        `TRANSIENT`, or any server error. A page or a lasting refusal is the same answer next
        time."""
        match self.outcome:
            case Unreached():
                return True
            case Unreadable(code=code):
                return code >= 500 or code in TRANSIENT
            case Page():
                return False
            case unreachable:
                assert_never(unreachable)


class _Text(HTMLParser):
    def __init__(self, main: bool = False) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[tuple[str, list[str]]] = []
        self._main = main
        self._skipping = 0
        self._layout = 0
        self._anchor: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._skipping += tag in _SKIP
        self._layout += tag in _LAYOUT
        if tag == "a" and (href := dict(attrs).get("href")):
            self._anchor = []
            self.links.append((href, self._anchor))

    def handle_endtag(self, tag: str) -> None:
        self._skipping -= tag in _SKIP and self._skipping > 0
        self._layout -= tag in _LAYOUT and self._layout > 0
        if tag == "a":
            self._anchor = None

    def handle_data(self, data: str) -> None:
        if self._skipping or not (text := data.strip()):
            return
        if self._anchor is not None:
            self._anchor.append(text)
        if not (self._main and self._layout):
            self.parts.append(text)


def text_of(html: str, main: bool = False) -> str:
    """A page's text. With `main`, its layout (navigation, header, footer, sidebars) is left out,
    and a passage the page repeats is kept once."""
    parser = _Text(main)
    parser.feed(html)
    parts = list(dict.fromkeys(parser.parts)) if main else parser.parts
    return " ".join(" ".join(parts).split())[:LIMIT]


def links_of(html: str, base: str) -> list[Link]:
    """The page's web links, made absolute against `base`, each once, with its anchor text."""
    parser = _Text()
    parser.feed(html)
    found: dict[str, str] = {}
    for href, words in parser.links:
        url = urljoin(base, href).split("#")[0]
        if urlsplit(url).scheme in ("http", "https"):
            found.setdefault(url, " ".join(words))
    return [Link(url=u, text=w) for u, w in list(found.items())[:LINKS]]


def _get(url: str) -> httpx.Response:
    return httpx.get(url, headers={"User-Agent": AGENT}, timeout=20, follow_redirects=True)


def retrieve(url: str, main: bool = False, get: Callable[[str], httpx.Response] = _get) -> Fetched:
    try:
        response = get(url)
    except httpx.HTTPError as err:
        return Fetched.of(url, Unreached(reason=type(err).__name__))
    kind = response.headers.get("content-type", "")
    if response.status_code != 200 or "html" not in kind:
        return Fetched.of(url, Unreadable(code=response.status_code, content_type=kind))
    final = str(response.url)
    text, links = text_of(response.text, main), links_of(response.text, final)
    return Fetched.of(url, Page(code=200, content_type=kind, text=text, final=final, links=links))


def lasting(op: DomainOp[Any], result: Any) -> bool:
    """Whether an answer may be kept in an op cache: anything but a transient fetch."""
    match result:
        case Fetched() if result.transient:
            return False
        case _:
            return True


def fetch(op: CallTool[Any]) -> Fetched:
    """The tool arm for `fetch`. A page asked before is answered by the run's op cache."""
    return retrieve(op.args["url"], op.args.get("main", False))


def host(url: str) -> str:
    name = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower()
    return name.removeprefix("www.")


class Hit(BaseModel):
    query: str
    title: str
    url: str
    description: str = ""
    """The search engine's snippet, where its API gives one."""
    age: str = ""
    """The page's date as the search engine gives it, where it gives one."""


class Searched(BaseModel):
    hits: list[Hit]
    tokens: int = 0
    """The model tokens the search cost; a search API prices by request instead."""
    kept: bool = False
    """Served from what is kept on disk, so this run spent nothing on it."""
    sent: int = 0
    """Requests this search sent to a search API; a query answered from disk sends none."""


class NotHeld(LookupError):
    """A run that may only replay kept searches met a query nobody kept."""


class SearchRefused(RuntimeError):
    """The search API answered a request with a status other than 200."""


BRAVE = "https://api.search.brave.com/res/v1/web/search"


def brave_url(query: str, count: int) -> str:
    """The request URL for `query`, which is also the name its response is kept under."""
    return BRAVE + "?" + urlencode({"q": query, "count": count, "text_decorations": "false"})


def _brave_get(url: str, key: str) -> httpx.Response:
    headers = {"Accept": "application/json", "X-Subscription-Token": key}
    return httpx.get(url, headers=headers, timeout=30)


@dataclass(frozen=True)
class BraveSearch:
    """The `search` tool on Brave's web search API: one request per query, its results in rank
    order. A request is charged before it is sent, so one lost in transit is counted too. A
    response is kept under its URL once its results read as hits, so a query is paid for once;
    the key travels in a header, so it is never part of what is kept."""

    key: str
    requests: TokenBudget
    """Requests sent, against the plan's allowance."""
    raw: Path
    count: int = 10
    get: Callable[[str, str], httpx.Response] = _brave_get

    def __post_init__(self) -> None:
        if not 1 <= self.count <= 20:
            raise ValueError(f"Brave returns 1 to 20 results a request, not {self.count}")

    def __call__(self, op: CallTool[Any]) -> Searched:
        answered = [self._hits(query) for query in op.args["queries"]]
        return Searched(
            hits=[hit for hits, _ in answered for hit in hits],
            kept=all(kept for _, kept in answered),
            sent=sum(not kept for _, kept in answered),
        )

    def _hits(self, query: str) -> tuple[list[Hit], bool]:
        url = brave_url(query, self.count)
        if (body := _kept(self.raw, url)) is not None:
            return _brave_hits(query, body), True
        self.requests.check()
        self.requests.add(1)
        response = self.get(url, self.key)
        if response.status_code != 200:
            raise SearchRefused(f"Brave answered {response.status_code} for {query!r}")
        body = response.json()
        hits = _brave_hits(query, body)
        limits = {k: v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit")}
        _keep(self.raw, url, {"url": url, "ratelimit": limits, **body})
        return hits, False


@dataclass(frozen=True)
class HeldSearch:
    """The `search` tool answering only from Brave responses already kept, so it spends nothing
    and needs no key."""

    raw: Path
    count: int = 10

    def __call__(self, op: CallTool[Any]) -> Searched:
        hits: list[Hit] = []
        for query in op.args["queries"]:
            if (body := _kept(self.raw, brave_url(query, self.count))) is None:
                raise NotHeld(f"no search is kept for {query!r}")
            hits += _brave_hits(query, body)
        return Searched(hits=hits, kept=True)


def _file(raw: Path, url: str) -> Path:
    return raw / (hashlib.sha256(url.encode()).hexdigest() + ".json")


def _kept(raw: Path, url: str) -> dict[str, Any] | None:
    kept = _file(raw, url)
    return json.loads(kept.read_text()) if kept.exists() else None


def _keep(raw: Path, url: str, body: dict[str, Any]) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=raw, suffix=".part", delete=False) as out:
        json.dump(body, out)
    os.replace(out.name, _file(raw, url))


def _brave_hits(query: str, body: dict[str, Any]) -> list[Hit]:
    results = body.get("web", {}).get("results", [])
    return [
        Hit(
            query=query,
            title=r["title"],
            url=r["url"],
            description=r.get("description", ""),
            age=r.get("page_age", "")[:10] or r.get("age", ""),
        )
        for r in results
    ]
