"""Read rendered markup by its RESULTS — attribute values and text — never by its bytes.

A renderer's escaping is its own business. `&quot;` and `&#34;` are the same character to a
browser, and an assertion that names one of them pins the ESCAPER rather than the guarantee: it
reddens when a renderer changes how it spells an escape and stays green when the value is wrong.
That is the wrong way round, and it is what happened when `cards.render_html` moved onto tdom —
one pin failed, and nothing about the page had changed.

So these parse. `HTMLParser` decodes entity references in attribute values and (with
`convert_charrefs`) in text, so what comes back is what a browser would see, and a test can say
what it means: *this attribute holds this value*, not *these bytes appear somewhere*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any


@dataclass
class Element:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    text: str = ""


class _Collect(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self._open: list[Element] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = Element(tag, {k: (v or "") for k, v in attrs})
        self.elements.append(element)
        self._open.append(element)

    def handle_endtag(self, tag: str) -> None:
        if self._open:
            self._open.pop()

    def handle_data(self, data: str) -> None:
        for element in self._open:
            element.text += data


def elements(markup: str, tag: str | None = None) -> list[Element]:
    """Every element, or every one with `tag`, in document order."""
    parser = _Collect()
    parser.feed(markup)
    parser.close()
    return [e for e in parser.elements if tag is None or e.tag == tag]


def one(markup: str, tag: str, **attrs: str) -> Element:
    """The single element matching `tag` and every named attribute — or a failure that says so."""
    found = [
        e for e in elements(markup, tag) if all(e.attrs.get(k) == v for k, v in attrs.items())
    ]
    assert len(found) == 1, f"expected exactly one <{tag} {attrs}>, found {len(found)}"
    return found[0]


def classes(markup: str) -> set[str]:
    """Every class token on the page — for asserting a variant landed, not where it landed."""
    return {c for e in elements(markup) for c in e.attrs.get("class", "").split()}


def json_attr(markup: str, tag: str, name: str) -> Any:
    """An attribute holding JSON, decoded — what `JSON.parse(el.dataset.…)` gets in the browser."""
    found = [e for e in elements(markup, tag) if name in e.attrs]
    assert len(found) == 1, f"expected one <{tag} {name}=…>, found {len(found)}"
    return json.loads(found[0].attrs[name])
