"""The wiki lint: links between pages resolve, and every page is reachable."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import wiki_lint  # noqa: E402


@pytest.fixture
def wiki(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway wiki: an index linking one concept page, which links back."""
    root = tmp_path / "wiki"
    (root / "concepts").mkdir(parents=True)
    (root / "index.md").write_text("# Index\n\n[concepts/a](concepts/a.md)\n")
    (root / "log.md").write_text("# Log\n")
    (root / "concepts" / "a.md").write_text("# A\n\nSee [index](../index.md#top).\n")
    for name, path in {
        "WIKI": root,
        "DEFERRED": root / "deferred.md",
        "TASKS": root / "tasks.md",
    }.items():
        monkeypatch.setattr(wiki_lint, name, path)
    return root


def test_a_relative_link_resolves_from_the_page_that_holds_it(wiki):
    assert wiki_lint.pages() == {"concepts/a": {"index"}, "index": {"concepts/a"}, "log": set()}
    assert wiki_lint.main() == 0


@pytest.mark.parametrize(
    "link",
    [
        '[i](../index.md "the catalog")',
        "[i](../index.md 'the catalog')",
        "[i](<../index.md>)",
        "[![b](x.png)](../index.md)",
        "[i][ref]\n\n[ref]: ../index.md",
    ],
)
def test_a_titled_or_bracketed_target_still_links(wiki, link):
    page = wiki / "concepts" / "b.md"
    page.write_text(link + "\n")
    assert wiki_lint.outbound(page) == {"index"}


def test_a_link_to_a_missing_page_is_dead(wiki, capsys):
    (wiki / "concepts" / "a.md").write_text("[gone](gone.md) [index](../index.md)\n")
    assert wiki_lint.main() == 1
    assert "DEAD   concepts/a links to concepts/gone" in capsys.readouterr().out


def test_a_page_nothing_links_to_is_an_orphan(wiki, capsys):
    (wiki / "concepts" / "b.md").write_text("[index](../index.md)\n")
    assert wiki_lint.main() == 1
    assert "ORPHAN concepts/b" in capsys.readouterr().out


@pytest.mark.parametrize(
    "line",
    [
        "`[b](b.md)`",
        "```\n[b](b.md)\n```",
        "~~~\n[b](b.md)\n~~~",
        "[site](https://example.com/b.md)",
        "[source](../../src/b.md)",
        "[section](#b)",
    ],
)
def test_code_urls_anchors_and_paths_outside_the_wiki_link_no_page(wiki, line):
    page = wiki / "concepts" / "b.md"
    page.write_text(line + "\n")
    assert wiki_lint.outbound(page) == set()


def test_a_bracketed_link_outside_code_fails(wiki, capsys):
    (wiki / "concepts" / "a.md").write_text("[[index]] and `[[code]]` and [index](../index.md)\n")
    assert wiki_lint.main() == 1
    assert "OLD    concepts/a.md: [[index]]" in capsys.readouterr().out
    assert wiki_lint.bracketed() == ["concepts/a.md: [[index]]"]
