"""Markdown as a t-string processor.

What an agent prints is markdown: a terminal renders it, a chat window renders it, and so does a
pipe through a renderer. A value spliced into it is text in that grammar, so a `|` in a value
splits a table cell, a `*` opens emphasis, and a newline then `#` starts a heading. `markdown`
renders a template's static text as the author wrote it, markup included, and each hole by what it
holds:

| a hole holding  | renders as                                                                    |
|-----------------|-------------------------------------------------------------------------------|
| a `Template`    | that template, rendered the same way, so the author's markup composes         |
| any other value | literal inline text, formatted by its spec, with markdown punctuation escaped |

A value's whitespace collapses to one space, so a newline in it cannot start a block.

`table` renders a GitHub-flavored table whose cells are templates or values. Inside a cell a static
`|` is escaped too, since no author means one to split the cell.
"""

from collections.abc import Iterable, Sequence
from string.templatelib import Interpolation, Template
from typing import Any, assert_never
from unicodedata import combining, east_asian_width

PUNCTUATION = frozenset("\\`*_[]<>|~#")
RULE = 3
"""The fewest dashes a column's separator cell takes."""
"""Characters escaped in a hole's text: each can open or close markdown here."""

_CONVERT = {"a": ascii, "r": repr, "s": str}

type Cell = Template | str | int | float


def markdown(template: Template) -> str:
    """The markdown `template` describes, each hole rendered as the table above says."""
    return _render(template, cell=False)


def table(head: Sequence[Cell], rows: Iterable[Sequence[Cell]]) -> str:
    """A table with `head` as its header row and one row per element of `rows`, each as wide.

    Each column is padded to its widest cell, counted in terminal columns, so the table reads
    aligned whether or not anything renders it."""
    width = len(head)
    lines = [_cells(head), *(_cells(row) for row in rows)]
    if bad := [i for i, row in enumerate(lines[1:]) if len(row) != width]:
        raise ValueError(f"table: rows {bad} are not {width} cells wide")
    widths = [max(RULE, *(_columns(line[i]) for line in lines)) for i in range(width)]
    rule = ["-" * w for w in widths]
    return "\n".join(_row(_padded(line, widths)) for line in [lines[0], rule, *lines[1:]])


def _cells(row: Sequence[Cell]) -> list[str]:
    return [_render(_as_template(value), cell=True) for value in row]


def _as_template(value: Cell) -> Template:
    match value:
        case Template():
            return value
        case str() | int() | float():
            return t"{value}"
        case unreachable:
            assert_never(unreachable)


def _padded(cells: Sequence[str], widths: Sequence[int]) -> list[str]:
    return [cell + " " * (w - _columns(cell)) for cell, w in zip(cells, widths, strict=True)]


def _columns(text: str) -> int:
    """The terminal columns `text` takes: two for a wide character, none for a combining mark."""
    return sum(0 if combining(c) else 2 if east_asian_width(c) in ("W", "F") else 1 for c in text)


def _row(cells: Sequence[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _render(template: Template, *, cell: bool) -> str:
    parts: list[str] = []
    for item in template:
        match item:
            case str() as text:
                parts.append(text.replace("|", "\\|") if cell else text)
            case Interpolation(value=Template() as nested):
                parts.append(_render(nested, cell=cell))
            case Interpolation() as hole:
                parts.append(_escaped(_text(hole)))
            case unreachable:
                assert_never(unreachable)
    return "".join(parts)


def _text(hole: Interpolation) -> str:
    value: Any = hole.value if hole.conversion is None else _CONVERT[hole.conversion](hole.value)
    return " ".join(format(value, hole.format_spec).split())


def _escaped(text: str) -> str:
    return "".join("\\" + c if c in PUNCTUATION else c for c in text)
