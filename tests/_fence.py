"""Reading a data block back: the label and the body, with the scan a reader would make.

A fence is `<left> <mark> data <label> <mark> <right>` … `<left> <mark> end <mark> <right>`, and
the bracket and mark BOTH vary with the level, so a block nested inside another is a different
alphabet rather than the same one repeated.
"""

import re

MARK = re.compile(
    r"\A(?P<left>\S\S) (?P<mark>\S+) data (?P<label>.*?) (?P=mark) (?P<right>\S\S)\n"
)


def parse(rendered: str) -> tuple[str, str]:
    """One data block's label and body.

    The scan stops at the FIRST close a reader would meet, and refuses anything after it."""
    found = MARK.match(rendered)
    assert found is not None, f"not a data block: {rendered!r}"
    closer = f"\n{found['left']} {found['mark']} end {found['mark']} {found['right']}"
    tail, closed, trailer = rendered[found.end() :].partition(closer)
    assert closed, rendered
    assert trailer == "", rendered
    return found["label"], tail


def body(rendered: str) -> str:
    return parse(rendered)[1]


def fence(rendered: str) -> tuple[str, str, str]:
    """The bracket and mark a block chose, as `(left, mark, right)`."""
    found = MARK.match(rendered)
    assert found is not None, rendered
    return found["left"], found["mark"], found["right"]


def mark(rendered: str) -> str:
    return fence(rendered)[1]


def after(prefix: str, rendered: str) -> tuple[str, str]:
    """The one block that follows a literal prefix, as `(label, body)`."""
    assert rendered.startswith(prefix), rendered
    return parse(rendered[len(prefix) :])


def within(prefix: str, rendered: str) -> tuple[str, str, str, str]:
    """The block following `prefix` inside a longer text, as `(label, mark, body, rest)`.

    `parse` refuses a trailer, which is what its callers want: nothing may follow the block they
    assert on. This is for the case where something does, and where splitting the text on the
    framing first is exactly the mistake the fence exists to survive. A body that imitates the
    surrounding prose puts that prose inside the block, so only the block's own close ends it.
    """
    assert rendered.startswith(prefix), rendered
    tail = rendered[len(prefix) :]
    found = MARK.match(tail)
    assert found is not None, f"not a data block: {tail!r}"
    closer = f"\n{found['left']} {found['mark']} end {found['mark']} {found['right']}"
    body, closed, rest = tail[found.end() :].partition(closer)
    assert closed, tail
    return found["label"], found["mark"], body, rest
