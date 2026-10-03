"""A web search query as a t-string processor.

A search query is a grammar: `site:`, a leading `-`, `OR` and a quoted phrase are operators, so a
value spliced in as text can change what is searched. A value from a page that reads
`2025 -site:example.org` would drop that host from the search. `search_query` renders the
template's static text as the author wrote it, operators included, and each hole by its spec:

| hole              | holds                     | renders as                                     |
|-------------------|---------------------------|------------------------------------------------|
| `{text}`          | a str or a number         | its words as plain terms, operators disarmed   |
| `{text:phrase}`   | a str or a number         | one quoted phrase                              |
| `{texts:either}`  | an iterable of them       | quoted phrases joined by `OR`                  |
| `{hosts:exclude}` | an iterable of host names | `-site:` before each host                      |

A term is disarmed by dropping its quotes and a leading `-` or `+`, splitting it at `:`, and
lowering an uppercase `OR` or `AND`. Any other spec, and any conversion, is refused.
"""

import re
from collections.abc import Iterable
from string.templatelib import Interpolation, Template
from typing import Any, assert_never

HOST = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*$")
"""A lowercase host name: dot-separated labels of letters, digits and inner hyphens."""
OPERATORS = {"OR", "AND"}


def search_query(template: Template) -> str:
    """The query `template` describes, its holes rendered as the table above says."""
    parts: list[str] = []
    for item in template:
        match item:
            case str() as text:
                parts.append(text)
            case Interpolation() as hole:
                parts.append(_hole(hole))
            case unreachable:
                assert_never(unreachable)
    return " ".join(" ".join(parts).split())


def _hole(hole: Interpolation) -> str:
    if hole.conversion is not None:
        raise ValueError(f"search_query: hole {hole.expression!r} takes no conversion")
    match hole.format_spec:
        case "":
            return " ".join(_terms(_text(hole.value, hole.expression)))
        case "phrase":
            return _phrase(_text(hole.value, hole.expression))
        case "either":
            phrases = [_phrase(_text(v, hole.expression)) for v in _many(hole)]
            return " OR ".join(phrase for phrase in phrases if phrase)
        case "exclude":
            return " ".join("-site:" + _host(v, hole.expression) for v in _many(hole))
        case spec:
            raise ValueError(
                f"search_query: hole {hole.expression!r} has an unknown spec {spec!r}"
            )


def _text(value: Any, expression: str) -> str:
    match value:
        case bool():
            raise ValueError(f"search_query: hole {expression!r} holds a bool, not text")
        case str() | int() | float():
            return str(value)
        case _:
            raise ValueError(
                f"search_query: hole {expression!r} holds {type(value).__name__}, not text"
            )


def _many(hole: Interpolation) -> Iterable[Any]:
    match hole.value:
        case str() | bytes():
            raise ValueError(
                f"search_query: hole {hole.expression!r} takes an iterable of values, not a string"
            )
        case Iterable() as values:
            return values
        case value:
            raise ValueError(
                f"search_query: hole {hole.expression!r} holds {type(value).__name__}, not values"
            )


def _terms(text: str) -> list[str]:
    terms = []
    for word in text.replace('"', " ").replace(":", " ").split():
        word = word.lstrip("-+")
        if word:
            terms.append(word.lower() if word in OPERATORS else word)
    return terms


def _phrase(text: str) -> str:
    words = text.replace('"', " ").split()
    return '"' + " ".join(words) + '"' if words else ""


def _host(value: Any, expression: str) -> str:
    match value:
        case str() as host if HOST.match(host):
            return host
        case _:
            raise ValueError(f"search_query: hole {expression!r} holds {value!r}, not a host name")
