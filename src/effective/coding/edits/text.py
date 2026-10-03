"""Rung 1, the text editor: exact replacements, several at once, each found in the original.

Every edit is located before any is applied, so no edit has to anticipate what another did to the
file. The located spans must be unique and disjoint, and the text between them is copied once, in
order. Two edits that touch are allowed; two that share a character are one edit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class TextEdit:
    """Replace `old`, which occurs exactly once in the original, with `new`."""

    old: str
    new: str


class EditRefused(ValueError):
    """An edit set that cannot be applied as written. The message is written for the model."""


@dataclass(frozen=True, slots=True)
class _Span:
    edit: int
    start: int
    end: int


def _locate(path: str, content: str, edits: Sequence[TextEdit]) -> list[_Span]:
    spans = []
    for index, edit in enumerate(edits):
        if not edit.old:
            raise EditRefused(f"edits[{index}].old is empty in {path}.")
        start = content.find(edit.old)
        if start < 0:
            raise EditRefused(
                f"Could not find edits[{index}].old in {path}. It must match exactly, including "
                f"whitespace and newlines."
            )
        occurrences, at = 0, start
        while at >= 0:
            occurrences, at = occurrences + 1, content.find(edit.old, at + 1)
        if occurrences > 1:
            raise EditRefused(
                f"Found {occurrences} occurrences of edits[{index}].old in {path}. It must be "
                f"unique; include more of the surrounding text."
            )
        spans.append(_Span(index, start, start + len(edit.old)))
    return sorted(spans, key=lambda span: span.start)


def apply_edits(path: str, content: str, edits: Sequence[TextEdit]) -> str:
    """`content` with every edit applied, or `EditRefused` naming the edit that cannot be.

    Refused: no edits, an empty `old`, an `old` that is absent or occurs more than once (counting
    overlapping occurrences), two edits whose spans overlap, and a set that changes nothing."""
    if not edits:
        raise EditRefused(f"No edits for {path}; give at least one replacement.")
    spans = _locate(path, content, edits)
    for before, after in pairwise(spans):
        if before.end > after.start:
            raise EditRefused(
                f"edits[{before.edit}] and edits[{after.edit}] overlap in {path}. Merge them into "
                f"one edit or target disjoint regions."
            )
    pieces, cursor = [], 0
    for span in spans:
        pieces += [content[cursor : span.start], edits[span.edit].new]
        cursor = span.end
    pieces.append(content[cursor:])
    edited = "".join(pieces)
    if edited == content:
        raise EditRefused(f"No changes made to {path}: the replacements give identical content.")
    return edited
