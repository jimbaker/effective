"""The text rung: every refusal by name, and generated edit sets over the repo's own files.

The generated half is where the rung's one property lives. Each edit is located in the original,
so the result cannot depend on the order the edits are listed in, even when a replacement contains
another edit's old text. The oracle splices by offset, from the end, which shares no code with
`apply_edits`."""

import random
from itertools import permutations
from pathlib import Path

import pytest

from effective.coding.edits.text import EditRefused, TextEdit, apply_edits

SOURCE = "def sub(a, b):\n    return a - b\n\n\ndef add(a, b):\n    return a + b\n"

REFUSED = [
    pytest.param([], "No edits", id="no-edits"),
    pytest.param([TextEdit("", "x")], r"edits\[0\]\.old is empty", id="empty-old"),
    pytest.param([TextEdit("a * b", "x")], r"Could not find edits\[0\]\.old", id="absent"),
    pytest.param([TextEdit("(a, b)", "x")], "Found 2 occurrences", id="ambiguous"),
    pytest.param(
        [TextEdit("return a - b", "x"), TextEdit("a - b\n\n", "y")],
        r"edits\[0\] and edits\[1\] overlap",
        id="overlapping",
    ),
    pytest.param([TextEdit("a - b", "a - b")], "No changes made", id="unchanged"),
]


@pytest.mark.parametrize(("edits", "message"), REFUSED)
def test_an_edit_set_that_cannot_apply_is_refused_by_name(edits, message):
    with pytest.raises(EditRefused, match=message):
        apply_edits("mod.py", SOURCE, edits)


def test_occurrences_are_counted_overlapping():
    with pytest.raises(EditRefused, match="Found 2 occurrences"):
        apply_edits("x.txt", "aaa", [TextEdit("aa", "b")])


def test_two_edits_that_touch_both_apply():
    edits = [TextEdit("a - ", "b - "), TextEdit("b\n\n", "a\n\n")]
    assert apply_edits("mod.py", SOURCE, edits).startswith("def sub(a, b):\n    return b - a\n")


def test_a_refusal_names_the_path():
    with pytest.raises(EditRefused, match=r"pkg/mod\.py"):
        apply_edits("pkg/mod.py", SOURCE, [TextEdit("absent", "x")])


CORPUS = [
    Path("src/effective/react.py"),
    Path("src/effective/machine/trampoline.py"),
    Path("src/effective/coding/tier.py"),
]


def occurrences(content: str, old: str) -> int:
    count, at = 0, content.find(old)
    while at >= 0:
        count, at = count + 1, content.find(old, at + 1)
    return count


def disjoint_unique_spans(content: str, rng: random.Random, want: int) -> list[tuple[int, int]]:
    """Up to `want` spans of one to three whole lines, each unique in `content`, none sharing a
    character with another."""
    starts = [0, *[i + 1 for i, char in enumerate(content) if char == "\n"][:-1]]
    spans: list[tuple[int, int]] = []
    for _ in range(200):
        if len(spans) == want:
            break
        first = rng.randrange(len(starts))
        last = min(first + rng.randint(1, 3), len(starts) - 1)
        start, end = starts[first], starts[last]
        if end <= start or occurrences(content, content[start:end]) != 1:
            continue
        if all(end <= s or start >= e for s, e in spans):
            spans.append((start, end))
    return spans


def spliced(content: str, spans: list[tuple[int, int]], news: list[str]) -> str:
    for (start, end), new in sorted(zip(spans, news, strict=True), reverse=True):
        content = content[:start] + new + content[end:]
    return content


CASES = [(path, seed) for path in CORPUS for seed in range(8)]


@pytest.mark.parametrize(("path", "seed"), CASES, ids=[f"{p.name}-{s}" for p, s in CASES])
def test_a_generated_edit_set_applies_by_offset_in_every_order(path, seed):
    content = path.read_text()
    rng = random.Random(seed)
    spans = disjoint_unique_spans(content, rng, want=rng.randint(2, 4))
    assert len(spans) >= 2, "the corpus should yield two disjoint unique spans"
    olds = [content[start:end] for start, end in spans]
    # A replacement that contains ANOTHER edit's old text: applied one after another, the later
    # edit would then find its text twice, or find the copy instead of the original.
    news = [olds[(i + 1) % len(olds)] + "# edited " + str(i) + "\n" for i in range(len(olds))]
    expected = spliced(content, spans, news)
    for order in permutations(range(len(olds))):
        edits = [TextEdit(olds[i], news[i]) for i in order]
        assert apply_edits(path.name, content, edits) == expected, order
