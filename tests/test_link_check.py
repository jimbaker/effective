"""The documentation link gate, proven on failures it can actually catch.

An instrument nobody has seen fail is not an instrument, so every rule here is tested by
MUTATION: build a corpus that satisfies the rule, break exactly one thing, and confirm the
right cell goes red.

The live-corpus tests are the gate itself; the synthetic ones fix the grammar and the
baseline's ratchet against a corpus that cannot drift.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from effective.lint import configured  # noqa: E402
from scripts import link_check as lc  # noqa: E402

WIKI_STUB = """# wiki

## Decisions

| ADR | Title | Status |
|---|---|---|
| 0003 | Permission cascade | *(no standalone file)* |
| 0021 | Dynamic graphs | Accepted |

## 5. Active work
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A miniature repo with the same shape as the real one — and a real git index.

    `resolves()` answers from `git ls-files`, so a corpus that is not a checkout has
    every citation in it dangling. `write(rel, text)` stages what it writes;
    `write(..., track=False)` is the untracked case, which has its own test.

    The module's path globals are re-pointed at the temp tree so nothing here
    can touch the real corpus.
    """

    # `-f` on every add, and the user's git config out of the environment: `build/` in a
    # global `core.excludesFile` makes `git add` REFUSE the path, and `init.templateDir`
    # would install their hooks into the corpus. A control the developer's machine can
    # change is not a control — which is the defect this whole module is about.
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
        )

    git("init", "-q")
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(lc, "REPO", tmp_path)
    # under scripts/, as in the real repo — a top-level prefix the grammar scans,
    # which is the whole point of the convergence test below
    monkeypatch.setattr(lc, "BASELINE", tmp_path / "scripts" / "link-check-baseline.txt")
    monkeypatch.setattr(lc, "WIKI", tmp_path / "wiki.md")
    # The real repo's generated-artifact declarations name real paths, and the pawl
    # fails on one nothing cites — which every synthetic corpus is. Start empty; the
    # two tests about GENERATED install their own.
    monkeypatch.setattr(lc, "GENERATED", {})
    (tmp_path / "wiki.md").write_text(WIKI_STUB)

    def write(rel: str, text: str, *, track: bool = True) -> Path:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if track:
            git("add", "-f", "--", rel)
        lc._tracked.cache_clear()
        return p

    return write


def paths(violations) -> set[str]:
    return {v.path for v in violations}


# --- the grammar ------------------------------------------------------------


def test_a_live_target_is_not_a_violation_and_a_dead_one_is(corpus):
    """The mutation baseline: identical citation, only the target differs."""
    corpus("docs/real.md", "# real")
    corpus("CLAUDE.md", "Read `docs/real.md`.")
    assert lc.scan() == []

    corpus("CLAUDE.md", "Read `docs/typo.md`.")
    assert paths(lc.scan()) == {"docs/typo.md"}


def test_an_untracked_target_does_not_resolve(corpus):
    """A gitignored artifact exists on a working machine and in no clone. A citation resolves
    against what a clone contains, or a local green predicts nothing about CI."""
    corpus("docs/generated.md", "# built, never committed", track=False)
    corpus("CLAUDE.md", "Read `docs/generated.md`.")
    assert paths(lc.scan()) == {"docs/generated.md"}


def test_a_declared_generated_artifact_resolves(corpus, monkeypatch):
    """The exemption an untracked artifact has to earn: a command that makes it."""
    monkeypatch.setattr(lc, "GENERATED", {"build/thing.json": "just build-thing"})
    corpus("CLAUDE.md", "The map is `build/thing.json`.")
    assert lc.scan() == []


def test_a_generated_declaration_is_stale_once_the_file_is_tracked(corpus, monkeypatch):
    """A tracked file needs no exemption, so the declaration goes stale and the table shrinks."""
    monkeypatch.setattr(lc, "GENERATED", {"build/thing.json": "just build-thing"})
    corpus("justfile", "build-thing:\n    echo hi\n")
    corpus("CLAUDE.md", "The map is `build/thing.json`.")
    assert lc.stale_generated(lc.cited_paths()) == []

    corpus("build/thing.json", "{}")
    assert [p for p, _ in lc.stale_generated(lc.cited_paths())] == ["build/thing.json"]


def test_a_generated_declaration_nobody_cites_is_stale(corpus, monkeypatch):
    """The pawl's second arm: an exemption for a citation that no longer exists."""
    monkeypatch.setattr(lc, "GENERATED", {"build/thing.json": "just build-thing"})
    corpus("CLAUDE.md", "no links here")
    assert [p for p, _ in lc.stale_generated(lc.cited_paths())] == ["build/thing.json"]


def test_shapes_that_are_not_links_are_never_scanned(corpus):
    """Globs, directory refs, prose-in-backticks, elisions. Each would be a false
    failure, and a gate with false failures gets disabled."""
    corpus(
        "CLAUDE.md",
        """
        `docs/*.md` and `wiki/**` are globs.
        `src/effective/` is a directory. `docs/` too.
        `no docs/x.md here` is prose. `docs/…-notes.md` is elided.
    """,
    )
    assert lc.scan() == []


# --- the ADR index ----------------------------------------------------------


def test_adr_resolves_against_the_wiki_rows_not_the_filesystem(corpus):
    """ADR-0003 has a ROW and deliberately no FILE, which a filesystem check would flag."""
    corpus("CLAUDE.md", "See ADR-0003 §2 and ADR-0021.")
    assert lc.scan() == []


def test_an_adr_with_no_row_is_a_violation(corpus):
    """The file exists and the index row does not: the row is what the gate and every reader
    resolve against."""
    corpus("docs/adr/0020-keys.md", "# adr 20")
    corpus("CLAUDE.md", "See ADR-0020.")
    assert [(v.path, v.kind) for v in lc.scan()] == [("ADR-0020", lc.Kind.ADR_UNKNOWN)]


# --- the baseline pawl ------------------------------------------------------


def _run(argv: list[str]) -> int:
    """Drive main() the way the justfile does — through argv, so the exit codes
    the recipe depends on are what is actually asserted."""
    old = sys.argv
    sys.argv = ["link_check.py", *argv]
    try:
        return lc.main()
    finally:
        sys.argv = old


def test_baseline_absorbs_existing_rot_then_fails_on_growth(corpus):
    corpus("CLAUDE.md", "Read `docs/old-rot.md`.")
    assert _run(["--init-baseline"]) == 0
    assert _run([]) == 0, "a baselined violation must not gate"

    corpus("CLAUDE.md", "Read `docs/old-rot.md` and `docs/new-rot.md`.")
    assert _run([]) == 1, "a NEW violation must gate even though rot is baselined"


def test_a_stale_baseline_entry_fails(corpus):
    """Without the stale rule, a fixed-then-re-broken pair is reabsorbed in silence."""
    corpus("CLAUDE.md", "Read `docs/gone.md`.")
    assert _run(["--init-baseline"]) == 0
    corpus("docs/gone.md", "# it came back")
    assert _run([]) == 1, "a baseline entry matching no violation must fail as stale"


def test_init_baseline_refuses_to_overwrite(corpus):
    corpus("CLAUDE.md", "Read `docs/x.md`.")
    assert _run(["--init-baseline"]) == 0
    assert _run(["--init-baseline"]) == 2, "bootstrap is one-time"


# --- the bootstrap, the declarations, and the wiki ---------------------------


def test_init_baseline_survives_committing_the_baseline(corpus):
    """`write_baseline` creates a file the corpus cites, untracked until the developer commits
    it. The citation resolves either way, so no entry goes stale on the commit."""
    corpus("CLAUDE.md", "The baseline lives at `scripts/link-check-baseline.txt`.")
    assert _run(["--init-baseline"]) == 0
    assert _run([]) == 0, "green before the baseline is committed"
    subprocess.run(
        ["git", "-C", str(lc.REPO), "add", "--", lc.BASELINE.relative_to(lc.REPO).as_posix()],
        check=True,
        capture_output=True,
    )
    lc._tracked.cache_clear()
    assert _run([]) == 0, "and green after — the bootstrap may not depend on staging order"


def test_a_generated_declaration_needs_a_producer_that_exists(corpus, monkeypatch):
    """A renamed recipe would leave the declaration excusing a path nothing produces."""
    monkeypatch.setattr(lc, "GENERATED", {"build/thing.json": "just no-such-recipe"})
    corpus("justfile", "build-thing:\n    echo hi\n")
    corpus("CLAUDE.md", "The map is `build/thing.json`.")
    assert [p for p, _ in lc.stale_generated(lc.cited_paths())] == ["build/thing.json"]


def test_a_wiki_citation_is_graded(corpus):
    """A repo-rooted `wiki/...` citation is a link here."""
    corpus("CLAUDE.md", "Read `wiki/concepts/nothing.md`.")
    assert paths(lc.scan()) == {"wiki/concepts/nothing.md"}


def test_a_relative_link_resolves_from_the_document_that_holds_it(corpus):
    corpus("src/a.py", "x = 1\n")
    corpus("docs/guide.md", "")
    corpus(
        "wiki/concepts/page.md",
        "[a](../../src/a.py) [b](../../src/b.py#L2) [guide](../../docs/guide.md) "
        "`[c](c.md)` [web](https://example.com/c.md) [here](#top)",
    )
    assert paths(lc.scan()) == {"src/b.py"}


def test_a_markdown_link_resolves_from_its_document_even_when_it_reads_repo_rooted(corpus):
    corpus("docs/guide.md", "")
    corpus("wiki/concepts/page.md", "[guide](docs/guide.md) [up](../../docs/guide.md)")
    assert paths(lc.scan()) == {"wiki/concepts/docs/guide.md"}


def test_a_directory_or_suffixless_target_is_graded(corpus):
    corpus("docs/guide.md", "")
    corpus(
        "wiki/page.md",
        '[dir](../docs/) [gone](../nope/) [bare](nope) [titled](../docs/guide.md "Guide")',
    )
    assert paths(lc.scan()) == {"nope", "wiki/nope"}


@pytest.mark.parametrize(
    ("link", "dead"),
    [
        ("[dead](missing.md 'title')", "wiki/missing.md"),
        ("[dead](missing.md (title))", "wiki/missing.md"),
        ("[dead](<missing file.md>)", "wiki/missing file.md"),
        ("[![image](../README.md)](missing.md)", "wiki/missing.md"),
        ("[dead][ref]\n\n[ref]: missing.md", "wiki/missing.md"),
        ("| a | b |\n|---|---|\n| `code | [dead](missing.md)` |", "wiki/missing.md"),
    ],
)
def test_every_link_a_renderer_draws_is_graded(corpus, link, dead):
    corpus("README.md", "")
    corpus("wiki/page.md", link)
    assert paths(lc.scan()) == {dead}


@pytest.mark.parametrize(
    "text",
    [
        "~~~\n[example](missing.md)\n~~~",
        "    [example](missing.md)",
        "[root](../)",
        "[r](../README%2Emd)",
    ],
)
def test_code_the_root_and_an_encoded_name_raise_nothing(corpus, text):
    corpus("README.md", "")
    corpus("wiki/page.md", "para\n\n" + text)
    assert paths(lc.scan()) == set()


# --- the live corpus: this IS the gate --------------------------------------


def test_the_live_corpus_has_no_new_and_no_stale_violations():
    violations = lc.scan()
    baseline = lc.load_baseline()
    new = [v for v in violations if v.pair not in baseline]
    stale = sorted(baseline.keys() - {v.pair for v in violations})
    assert not new, f"{len(new)} new dead documentation link(s): {new[:5]}"
    assert not stale, f"{len(stale)} stale baseline entr(ies) — delete them: {stale[:5]}"


def test_every_generated_declaration_is_untracked_and_still_cited():
    """The live half of the pawl: the gate's own hole, measured on this corpus."""
    assert lc.stale_generated(lc.cited_paths()) == []


def test_every_adr_citation_resolves_against_the_wiki_index():
    """A document's ADR number resolves against a row of the wiki's `## Decisions` table, which is
    the one index; a file with no row is unknown."""
    unknown = [v for v in lc.scan() if v.kind is lc.Kind.ADR_UNKNOWN]
    assert not unknown, f"ADR citations with no `## Decisions` row: {unknown}"


# the ADR checker and its tests name ADRs as their subject; a tree that ships more declares its
# own exemptions, each a path prefix, in `[tool.effective.lint].adr_citation_exempt`
_ADR_EXEMPT = (
    "scripts/link_check.py",
    "scripts/doc_inventory.py",
    "tests/test_link_check.py",
    *(configured("adr_citation_exempt") or ()),
)


def test_code_tests_and_models_cite_no_adr():
    """An ADR's sections move when it is rewritten, so code states its invariant in its own words.
    Scanned: every tracked file under the code, test, example, model and script trees."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "src", "tests", "examples", "formal", "scripts"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    tracked = [path for path in tracked if path]
    assert tracked, "no tracked files: this test needs a git checkout"
    citing = [
        f"{path}:{n}"
        for path in tracked
        if not path.startswith(_ADR_EXEMPT)
        for n, line in enumerate((_ROOT / path).read_text(errors="replace").splitlines(), 1)
        if lc.ADR_REF_RE.search(line)
    ]
    assert not citing, f"{len(citing)} ADR citations in code: {citing[:20]}"


def test_the_maintained_layer_is_scanned():
    """`docs/` claims to be current, so it is scanned; third-party input is not ours to grade."""
    rels = {p.relative_to(lc.REPO).as_posix() for p in lc.sources()}
    assert "docs/effective-101.md" in rels
    assert "docs/README.md" in rels
    assert not any(p.startswith("external-docs/") for p in rels), (
        "third-party input is not ours to grade"
    )


def test_the_wikilink_resolver_knows_the_maintained_layer():
    """Anything in docs/ is visible to the resolver: without the docs/ stems, a document
    promoted into docs/ would go invisible the moment it moved."""
    from scripts.doc_inventory import resolve_wikilinks

    assert resolve_wikilinks({"src": ["effective-101", "first-workflow"]})["dangling"] == []


def test_the_gate_is_actually_reading_the_corpus():
    """Anti-vacuity, asserted on the scan's coverage rather than on a count the work moves.

    Every test above passes vacuously if `sources()` finds nothing (a wrong REPO, a broken glob).
    So: every root family is reached.
    """
    rels = {p.relative_to(lc.REPO).as_posix() for p in lc.sources()}

    assert "CLAUDE.md" in rels, "the root instructions must be scanned"
    for family in ("docs/", "wiki/"):
        assert any(r.startswith(family) for r in rels), f"{family} must be scanned"


def test_an_out_of_range_section_pointer_is_flagged(corpus):
    """A `see §N` naming a section the document lacks is caught.

    A pointer to a section that EXISTS and is the wrong one is not, and stays a review-time
    check; the next test pins that limit.
    """
    corpus("docs/thing.md", "## 1. One\n\nSee §9 for details.\n\n## 2. Two\n")
    assert [(v.path, v.kind) for v in lc.scan()] == [("§9", lc.Kind.SECTION)]


def test_a_wrong_but_existing_section_pointer_is_NOT_flagged(corpus):
    """The boundary of an automatable check: which section was meant is not in the text."""
    corpus("docs/thing.md", "## 1. One\n\nSee §2 — exists, but is the wrong one.\n\n## 2. Two\n")
    assert lc.scan() == [], "no automatable rule can know which section was MEANT"


def test_a_subsection_pointer_is_validated_by_its_FULL_dotted_path(corpus):
    """Checking only the major number would let `see §1.99` pass under an existing §1."""
    corpus("docs/thing.md", "## 1. One\n\nSee §1.99 here.\n")
    assert [v.path for v in lc.scan()] == ["§1.99"]

    corpus("docs/thing.md", "## 1. One\n\n### 1.2 Sub\n\nSee §1.2 here.\n")
    assert lc.scan() == [], "a real subsection must resolve"


def test_headings_deeper_than_three_hashes_are_seen(corpus):
    """A correct pointer to a `####` heading resolves: a gate that fires on correct prose is how
    gates get disabled."""
    corpus("docs/thing.md", "## 1. One\n\n#### 1.2 Deep\n\nSee §1.2 here.\n")
    assert lc.scan() == []
    # a major-numbered `####` heading, pinned on its own rather than jointly with the half above
    corpus("docs/thing.md", "## 1. One\n\n#### 9. Deep\n\nSee §9 here.\n")
    assert lc.scan() == []
