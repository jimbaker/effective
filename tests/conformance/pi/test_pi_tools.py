"""Pi's tool tests, translated onto `examples.coder`'s tools. Provenance and scope: `SOURCE.md`.

ROLE: conformance. The cases were chosen by another project's suite, so each row states
what Pi's case asserts, and a disagreement is either a defect here or a row in `SOURCE.md`.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

import pytest

from effective.coding.edits.text import EditRefused
from effective.coding.tier import IMAGE, image_available
from effective.combinators import Level
from effective.react import ToolLog, ToolRequest, typed_act
from examples.coder.tools import (
    TOOLS,
    BashArgs,
    Changed,
    EditArgs,
    Ran,
    ReadArgs,
    Refused,
    Replacement,
    Viewed,
    WriteArgs,
    _tail,
    bash,
    edit,
    read,
    write,
)

needs_image = pytest.mark.skipif(not image_available(), reason=f"{IMAGE} is not built")


def numbered(count: int, line: str = "Line {}") -> str:
    return "\n".join(line.format(i + 1) for i in range(count))


@dataclass(frozen=True)
class Shows:
    """A read that succeeds: its exact text when the case states one, what the text holds and
    lacks otherwise, and a pattern its notice matches. An empty pattern means the read shows the
    whole selection and says nothing."""

    text: str | None = None
    has: tuple[str, ...] = ()
    lacks: tuple[str, ...] = ()
    notice: str = ""


@dataclass(frozen=True)
class Refuses:
    error: type[Exception]
    match: str


@dataclass(frozen=True)
class Leaves:
    """An edit or write that succeeds, leaving `content` at `path`."""

    path: str
    content: str


type Expect = Shows | Refuses | Leaves

HELLO = {"edit-test.txt": "Hello, world!"}
HUNDRED = {"f.txt": numbered(100)}

READS = {
    "should read file contents that fit within limits": (
        {"test.txt": "Hello, world!\nLine 2\nLine 3"},
        ReadArgs(path="test.txt"),
        Shows(text="Hello, world!\nLine 2\nLine 3"),
    ),
    "should handle non-existent files": (
        HELLO,
        ReadArgs(path="nonexistent.txt"),
        Refuses(Refused, "no file nonexistent.txt"),
    ),
    "should truncate files exceeding line limit": (
        {"large.txt": numbered(2500)},
        ReadArgs(path="large.txt"),
        Shows(
            has=("Line 1\n", "Line 2000\n"),
            lacks=("Line 2001",),
            notice=r"Showing lines 1-2000 of 2500\. Use offset=2001 to continue\.",
        ),
    ),
    "should truncate when byte limit exceeded": (
        {"large-bytes.txt": numbered(500, "Line {}: " + "x" * 200)},
        ReadArgs(path="large-bytes.txt"),
        Shows(
            has=("Line 1:",),
            lacks=("Line 500:",),
            notice=r"Showing lines 1-\d+ of 500 \(50 KB limit\)\. Use offset=\d+ to continue\.",
        ),
    ),
    "should handle offset parameter": (
        HUNDRED,
        ReadArgs(path="f.txt", offset=51),
        Shows(text=numbered(100)[numbered(100).index("Line 51") :]),
    ),
    "should handle limit parameter": (
        HUNDRED,
        ReadArgs(path="f.txt", limit=10),
        Shows(text=numbered(10) + "\n", notice=r"offset=11 "),
    ),
    "should handle offset + limit together": (
        HUNDRED,
        ReadArgs(path="f.txt", offset=41, limit=20),
        Shows(text="".join(f"Line {i}\n" for i in range(41, 61)), notice=r"offset=61 "),
    ),
    "should show error when offset is beyond file length": (
        {"short.txt": "Line 1\nLine 2\nLine 3"},
        ReadArgs(path="short.txt", offset=100),
        Refuses(Refused, "offset 100 .* has 3 lines"),
    ),
    "should treat files with image extension but non-image content as text": (
        {"not-an-image.png": "definitely not a png"},
        ReadArgs(path="not-an-image.png"),
        Shows(text="definitely not a png"),
    ),
}


def edited(tree: dict[str, str], path: str, *pairs: tuple[str, str]) -> tuple[Any, ...]:
    return tree, EditArgs(path=path, edits=[Replacement(old_text=o, new_text=n) for o, n in pairs])


CHANGES = {
    "should write file contents": (
        {},
        WriteArgs(path="write-test.txt", content="Test content"),
        Leaves("write-test.txt", "Test content"),
    ),
    "should create parent directories": (
        {},
        WriteArgs(path="nested/dir/test.txt", content="Nested content"),
        Leaves("nested/dir/test.txt", "Nested content"),
    ),
    "should replace text in file": (
        *edited(HELLO, "edit-test.txt", ("world", "testing")),
        Leaves("edit-test.txt", "Hello, testing!"),
    ),
    "should fail if text not found": (
        *edited(HELLO, "edit-test.txt", ("nonexistent", "testing")),
        Refuses(EditRefused, "Could not find"),
    ),
    "should include ENOENT when the edit target does not exist": (
        *edited(HELLO, "missing.txt", ("hello", "world")),
        Refuses(Refused, "no file missing.txt"),
    ),
    "should fail if text appears multiple times": (
        *edited({"e.txt": "foo foo foo"}, "e.txt", ("foo", "bar")),
        Refuses(EditRefused, "Found 3 occurrences"),
    ),
    "should replace multiple disjoint regions in one call": (
        *edited(
            {"m.txt": "alpha\nbeta\ngamma\ndelta\n"},
            "m.txt",
            ("alpha\n", "ALPHA\n"),
            ("gamma\n", "GAMMA\n"),
        ),
        Leaves("m.txt", "ALPHA\nbeta\nGAMMA\ndelta\n"),
    ),
    "should match edits against the original file, not incrementally": (
        *edited({"o.txt": "foo\nbar\nbaz\n"}, "o.txt", ("foo\n", "foo bar\n"), ("bar\n", "BAR\n")),
        Leaves("o.txt", "foo bar\nBAR\nbaz\n"),
    ),
    "should fail when edits is empty": (
        *edited({"x.txt": "hello\nworld\n"}, "x.txt"),
        Refuses(EditRefused, "No edits"),
    ),
    "should fail when multi-edit regions overlap": (
        *edited(
            {"v.txt": "one\ntwo\nthree\n"},
            "v.txt",
            ("one\ntwo\n", "ONE\nTWO\n"),
            ("two\nthree\n", "TWO\nTHREE\n"),
        ),
        Refuses(EditRefused, "overlap"),
    ),
    "should not partially apply edits when one edit fails": (
        *edited(
            {"p.txt": "alpha\nbeta\ngamma\n"},
            "p.txt",
            ("alpha\n", "ALPHA\n"),
            ("missing\n", "MISSING\n"),
        ),
        Refuses(EditRefused, "Could not find"),
    ),
    "should preserve LF line endings for LF files": (
        *edited({"l.txt": "first\nsecond\nthird\n"}, "l.txt", ("second\n", "REPLACED\n")),
        Leaves("l.txt", "first\nREPLACED\nthird\n"),
    ),
    "should prefer exact match over fuzzy match": (
        *edited(
            {"x.js": "const x = 'exact';\nconst y = 'other';\n"},
            "x.js",
            ("const x = 'exact';", "const x = 'changed';"),
        ),
        Leaves("x.js", "const x = 'changed';\nconst y = 'other';\n"),
    ),
    "should still fail when text is not found even with fuzzy matching": (
        *edited(
            {"n.txt": "completely different content\n"},
            "n.txt",
            ("this does not exist", "replacement"),
        ),
        Refuses(EditRefused, "Could not find"),
    ),
}

LINE_ENDINGS = pytest.mark.xfail(
    strict=True,
    reason="the coder's edits match bytes exactly; line-ending matching is not implemented",
)
CRLF = {
    "should match LF oldText against CRLF file content": (
        *edited(
            {"c.txt": "line one\r\nline two\r\nline three\r\n"},
            "c.txt",
            ("line two\n", "replaced line\n"),
        ),
        Leaves("c.txt", "line one\r\nreplaced line\r\nline three\r\n"),
    ),
    "should preserve CRLF line endings after edit": (
        *edited({"c.txt": "first\r\nsecond\r\nthird\r\n"}, "c.txt", ("second\n", "REPLACED\n")),
        Leaves("c.txt", "first\r\nREPLACED\r\nthird\r\n"),
    ),
    "should detect duplicates across CRLF/LF variants": (
        *edited(
            {"c.txt": "hello\r\nworld\r\n---\r\nhello\nworld\n"},
            "c.txt",
            ("hello\nworld\n", "replaced\n"),
        ),
        Refuses(EditRefused, "Found 2 occurrences"),
    ),
    "should preserve UTF-8 BOM after edit": (
        *edited(
            {"c.txt": "\ufefffirst\r\nsecond\r\nthird\r\n"}, "c.txt", ("second\n", "REPLACED\n")
        ),
        Leaves("c.txt", "\ufefffirst\r\nREPLACED\r\nthird\r\n"),
    ),
    "should preserve CRLF line endings and BOM in multi-edit mode": (
        *edited(
            {"c.txt": "\ufefffirst\r\nsecond\r\nthird\r\nfourth\r\n"},
            "c.txt",
            ("second\n", "SECOND\n"),
            ("fourth\n", "FOURTH\n"),
        ),
        Leaves("c.txt", "\ufefffirst\r\nSECOND\r\nthird\r\nFOURTH\r\n"),
    ),
}


def settle(call: Any, tree: dict[str, str], args: Any, expect: Expect) -> None:
    before = dict(tree)
    match expect:
        case Refuses(error=error, match=pattern):
            with pytest.raises(error, match=pattern):
                call(tree, args)
            assert tree == before
        case Shows(text=text, has=has, lacks=lacks, notice=notice):
            viewed = call(tree, args)
            assert isinstance(viewed, Viewed)
            if text is not None:
                assert viewed.text == text
            assert all(part in viewed.text for part in has)
            assert not any(part in viewed.text for part in lacks)
            if notice:
                assert re.search(notice, viewed.notice), viewed.notice
            else:
                assert viewed.notice == ""
        case Leaves(path=path, content=content):
            changed = call(tree, args)
            assert isinstance(changed, Changed)
            assert (changed.path, changed.tree[path]) == (path, content)
            assert tree == before
        case unreachable:
            assert_never(unreachable)


def change(tree: dict[str, str], args: WriteArgs | EditArgs) -> Changed:
    match args:
        case WriteArgs():
            return write(tree, args)
        case EditArgs():
            return edit(tree, args)
        case unreachable:
            assert_never(unreachable)


@pytest.mark.conformance
@pytest.mark.parametrize(("tree", "args", "expect"), READS.values(), ids=READS.keys())
def test_read(tree: dict[str, str], args: ReadArgs, expect: Expect) -> None:
    settle(read, tree, args, expect)


@pytest.mark.conformance
@pytest.mark.parametrize(
    ("tree", "args", "expect"),
    [
        *(pytest.param(*row, id=title) for title, row in CHANGES.items()),
        *(pytest.param(*row, id=title, marks=LINE_ENDINGS) for title, row in CRLF.items()),
    ],
)
def test_edit_and_write(tree: dict[str, str], args: WriteArgs | EditArgs, expect: Expect) -> None:
    settle(change, tree, args, expect)


@pytest.mark.conformance
def test_keeps_legacy_fields_out_of_the_public_schema() -> None:
    assert {"oldText", "newText"}.isdisjoint(EditArgs.model_json_schema()["properties"])


LEGACY = {
    "folds top-level oldText/newText into edits": {
        "path": "file.txt",
        "oldText": "before",
        "newText": "after",
    },
    "parses edits from a JSON string": {
        "path": "file.txt",
        "edits": '[{"old_text": "a", "new_text": "b"}]',
    },
}


@pytest.mark.conformance
@pytest.mark.parametrize("given", LEGACY.values(), ids=LEGACY.keys())
def test_a_legacy_spelling_is_bad_arguments(given: dict[str, Any]) -> None:
    act = typed_act(TOOLS, ToolLog())
    loop = act(ToolRequest(name="edit", args=given), Level(0, "m", final=False))
    with pytest.raises(StopIteration) as stopped:
        next(loop)
    assert stopped.value.value.content.startswith("[bad arguments for edit]")


@pytest.mark.conformance
def test_should_not_count_a_trailing_newline_as_an_extra_truncated_bash_output_line() -> None:
    shown, notice = _tail(numbered(4000, "line-{:04}") + "\n")
    assert "line-2001" in shown
    assert "line-4000" in shown
    assert "line-2000" not in shown
    assert "of 4000 lines" in notice


BASH = {
    "should execute simple commands": ("echo 'test output'", None, 0, "test output\n"),
    "should handle command errors": ("exit 1", None, 1, ""),
    "should respect timeout": ("sleep 30", 1, 124, None),
    "should decode UTF-8 characters split across output chunks": (
        "printf '\\342\\202'; printf '\\254\\n'",
        None,
        0,
        "€\n",
    ),
}


@needs_image
@pytest.mark.conformance
@pytest.mark.parametrize(
    ("command", "timeout", "exit_code", "output"), BASH.values(), ids=BASH.keys()
)
def test_bash(command: str, timeout: int | None, exit_code: int, output: str | None) -> None:
    ran = bash({"a.txt": "a"}, BashArgs(command=command, timeout=timeout))
    assert isinstance(ran, Ran)
    assert ran.exit_code == exit_code
    if output is not None:
        assert ran.output == output


TRANSLATED_BY_NAME = {
    "keeps legacy fields out of the public schema": (
        test_keeps_legacy_fields_out_of_the_public_schema
    ),
    "should not count a trailing newline as an extra truncated bash output line": (
        test_should_not_count_a_trailing_newline_as_an_extra_truncated_bash_output_line
    ),
}

HERE = Path(__file__).parent


def upstream() -> list[tuple[str, str]]:
    """Each Pi case at the pinned commit, as its describe block and its title."""
    lines = (HERE / "titles.txt").read_text().splitlines()
    cases = [line.split("\t") for line in lines if not line.startswith("#")]
    return [(block, title) for _file, block, title in cases]


@pytest.mark.conformance
def test_every_upstream_case_is_translated_or_named() -> None:
    source = Path(__file__).read_text()
    page = (HERE / "SOURCE.md").read_text()
    unaccounted = [
        title
        for block, title in upstream()
        if f'"{title}"' not in source
        and title not in TRANSLATED_BY_NAME
        and f"`{title}`" not in page
        and f"`{block}` block" not in page
    ]
    assert unaccounted == []


@pytest.mark.conformance
def test_the_page_names_only_upstream_cases() -> None:
    """Every name in the first column of the page's tables is a block or a title at the pin."""
    known = {part for case in upstream() for part in case}
    rows = [
        line for line in (HERE / "SOURCE.md").read_text().splitlines() if line.startswith("| ")
    ]
    named = {name for row in rows for name in re.findall(r"`([^`]+)`", row.split("|")[1])}
    assert named - known == set()
