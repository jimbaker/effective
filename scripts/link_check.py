"""Documentation link gate: a dangling doc link fails like a broken test.

The scanned documents are the maintained layer, current by contract: `README.md`, `CLAUDE.md`,
`docs/` and `wiki/`. Each cites the tree three ways, and each form is resolved:

| form                         | resolves against                                       |
|------------------------------|--------------------------------------------------------|
| a repo-rooted path           | the tracked tree, or a producer declared in GENERATED  |
| `ADR-NNNN`                   | a row of the wiki's `## Decisions` table               |
| `see §N` inside a document   | a numbered heading of that same document               |

**A baseline with a ratchet, so pre-existing rot can be recorded once and the gate fails only
on growth.** A baseline entry matching no current violation is an ERROR ("stale: delete it"),
so a fixed-then-re-broken pair cannot be reabsorbed in silence. Nothing appends to the
baseline after the bootstrap: a new break is fixed, never recorded.

Usage:
    uv run python scripts/link_check.py                # the gate (just docs-check)
    uv run python scripts/link_check.py --verbose      # every violation, not a summary
    uv run python scripts/link_check.py --init-baseline    # bootstrap, once
    uv run python scripts/link_check.py --prune-stale      # drop entries that no longer fail
"""

import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "scripts" / "link-check-baseline.txt"
# The gate's own artifact resolves by construction. The bootstrap creates it, so it is untracked
# in the one run that cites it, and recording that citation would leave an entry that goes stale
# the moment the developer commits the file.
BASELINE_PATH = "scripts/link-check-baseline.txt"
WIKI = REPO / "wiki" / "index.md"  # the catalog; its `## Decisions` table is the ONE ADR index

# --- the link grammar (shared: doc_inventory.py imports from here) ----------

# A link is a backticked token or a markdown-link target. Bare prose paths are
# deliberately NOT scanned: a sentence naming a directory in passing is not a link,
# and the false-positive cost exceeds the benefit.
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
MDLINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# a code anchor: path:line, the grammar doc_inventory.py shares
PATHLINE_RE = re.compile(r"([\w./-]+\.(?:py|sql|lean|qnt|toml|sh|ts|js|yml|yaml)):(\d+)")

# an ADR is cited BY NUMBER, never by path: the form that survives a move
ADR_REF_RE = re.compile(r"\bADR-(\d{4})\b")
# An INTRA-document section pointer, and deliberately only the `see §N` form: a bare `§N` usually
# names ANOTHER document's section (a page citing an ADR's §9), which this document cannot resolve.
SEE_SECTION_RE = re.compile(r"see §(\d+(?:\.\d+)*)", re.IGNORECASE)
HEADING_NUM_RE = re.compile(r"^#{2,6} (\d+(?:\.\d+)*)[.\u00a0 ]", re.MULTILINE)
FENCE_BLOCK_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)

# the `## Decisions` table's rows: | 0021 | Title | Status | … |
ADR_ROW_RE = re.compile(r"^\|\s*(\d{4})\s*\|", re.MULTILINE)
ADR_SECTION_RE = re.compile(r"^## Decisions$(.*?)^## ", re.MULTILINE | re.DOTALL)

# repo-rooted means: starts with a directory that exists at the repo root
TOP_LEVEL = (
    "src/",
    "tests/",
    "scripts/",
    "docs/",
    "examples/",
    "infra/",
    "formal/",
    "migrations/",
    "build/",
    "wiki/",
)
SUFFIXED = re.compile(
    r"\.(md|html|pdf|json|jsonl|py|sql|lean|qnt|toml|sh|ts|js|yml|yaml|txt"
    r"|docx|csv|svg|css|xml|ini|cfg|lock)$"
)
# trailing locators a reader adds to a path: :12, :12-40, ::test_name, #anchor
LOCATOR_RE = re.compile(r"(::[\w:.\[\]-]+|:\d+(?:-\d+)?)$")


class Kind(StrEnum):
    GONE = "gone"  # nothing by that name in the tracked tree
    ADR_UNKNOWN = "adr"  # ADR-00NN with no row in the `## Decisions` table
    SECTION = "section"  # `see §N` pointing at a section this document lacks


@dataclass(frozen=True)
class Violation:
    source: str  # repo-relative path of the citing document
    path: str  # the dead link, as written (locator stripped)
    kind: Kind

    @property
    def pair(self) -> tuple[str, str]:
        """Baseline identity. Keyed by SOURCE, so a move orphans its entries."""
        return (self.source, self.path)


def _exempt(token: str) -> bool:
    """Shapes that look like paths but are not links to a file in this repo.

    Measured against the whole surface: globs and placeholders, directory
    references, prose that happened to fall inside backticks, and elided paths.
    """
    return (
        any(ch in token for ch in "*?<>{}|")
        or token.endswith("/")
        or " " in token
        or "…" in token
        or ".." in token
    )


# Paths this repo PRODUCES: cited legitimately, gitignored by design, and absent from any fresh
# clone. `_exempt` grades a SHAPE and this grades an IDENTITY, which is why it is a table. Each
# entry names the command that makes the file, because a citation names something reproducible.
#
# Held to the baseline's ratchet. An entry is stale when its file becomes tracked, when nothing
# cites it, or when its producer names no justfile recipe: a renamed recipe would otherwise leave
# the declaration excusing a path nothing makes.
GENERATED: dict[str, str] = {
    "build/key-registry.json": "just key-registry",
}


@cache
def _tracked(repo: Path) -> frozenset[str]:
    """Every path in git's index, as repo-relative posix strings.

    Cached per repo: no mode here changes the index, so one `git ls-files` serves a whole
    scan. Anything that DOES change it, such as a corpus built a file at a time, owes a
    `_tracked.cache_clear()`.
    """
    # `git -C` does NOT override GIT_DIR, and a git hook sets exactly that, so a
    # pre-commit hook running this gate would read another index and report every citation
    # dangling. Name the repository instead of inheriting it.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")
    }
    try:
        out = subprocess.run(
            ["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}", "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError:
        raise SystemExit(
            f"link_check: no `git` on PATH; this gate resolves citations "
            f"against the tracked tree of {repo}."
        ) from None
    if out.returncode != 0:
        raise SystemExit(
            f"link_check: `git ls-files` failed in {repo}. This gate resolves citations "
            f"against the TRACKED tree, so it has to run inside a checkout.\n"
            f"{out.stderr.strip()}"
        )
    return frozenset(path for path in out.stdout.split("\0") if path)


def links(text: str) -> Iterator[str]:
    """Every repo-rooted-looking link target in a document, locator stripped."""
    for match_re in (BACKTICK_RE, MDLINK_RE):
        for m in match_re.finditer(text):
            token = m.group(1).split("#", 1)[0]
            if not token.startswith(TOP_LEVEL) or _exempt(token):
                continue
            if SUFFIXED.search(path := LOCATOR_RE.sub("", token)):
                yield path


def adr_index() -> set[str]:
    """ADR numbers with a row in the wiki's `## Decisions` table, the ONE index.

    Never the filesystem: an ADR may have a row and no standalone file. Without the table there
    is no ADR to cite, so every `ADR-NNNN` citation is unknown.
    """
    if not WIKI.exists() or not (s := ADR_SECTION_RE.search(WIKI.read_text(encoding="utf-8"))):
        return set()
    return {m.group(1) for m in ADR_ROW_RE.finditer(s.group(1))}


def resolves(path: str) -> bool:
    """Repo-rooted and TRACKED, or a producer declared in GENERATED.

    The verdict is a function of what a clone contains. A working tree also holds build output,
    hook output and everything gitignored, so a file that exists only locally resolves nowhere
    else; a file deleted but not `git rm`-ed still resolves, here and in CI alike.
    """
    return path in GENERATED or path == BASELINE_PATH or path in _tracked(REPO)


def cited_paths() -> Counter[str]:
    """How many times each repo-rooted target is named across the scanned corpus."""
    return Counter(
        path
        for src in sources()
        for path in links(src.read_text(encoding="utf-8", errors="replace"))
    )


def _producer_missing(producer: str) -> bool:
    """A declared producer that names nothing: an absent recipe, or an untracked file.

    An existence check for a recipe name at column 0. `preflight.parse_recipes` parses the
    justfile properly, but `scripts/` is not a package, so importing it would work when this
    module runs as a script and fail when the tests import it as `scripts.link_check`.
    """
    if producer.startswith("just "):
        # Read at call time: REPO is what the tests re-point, so a module constant bound at
        # import would read the real justfile while everything else read the test corpus.
        justfile = REPO / "justfile"
        if not justfile.exists():
            return True
        name = re.escape(producer.split()[1])
        return not re.search(rf"^{name}(\s[^:]*)?:", justfile.read_text(encoding="utf-8"), re.M)
    return producer not in _tracked(REPO)


def stale_generated(cited: Mapping[str, int]) -> list[tuple[str, str]]:
    """GENERATED entries that no longer earn their exemption, each with the reason."""
    tracked = _tracked(REPO)
    out: list[tuple[str, str]] = []
    for path, producer in sorted(GENERATED.items()):
        if path in tracked:
            out.append((path, "now tracked, so the exemption is what is stale"))
        elif not cited.get(path):
            out.append((path, "no scanned document cites it any more"))
        elif _producer_missing(producer):
            out.append((path, f"{producer!r} names no recipe and no tracked file"))
    return out


def unknown_adrs(text: str, adrs: set[str]) -> Iterator[str]:
    """`ADR-00NN` citations with no row in the wiki's ADR index."""
    for m in ADR_REF_RE.finditer(text):
        if m.group(1) not in adrs:
            yield f"ADR-{m.group(1)}"


def dangling_sections(text: str) -> Iterator[str]:
    """`see §N` pointers that name no section of THIS document.

    Caught: a `see §N` or `see §N.M` whose full dotted path matches no numbered heading at any
    level (`##` through `######`), as left behind when a section is deleted or renumbered.
    Uncaught: a pointer to a section that EXISTS and is the wrong one, which stays a review-time
    check.
    """
    # Fenced blocks are stripped first: a heading-shaped line quoted inside a fence would count
    # as a real section, masking a dangling pointer or inventing a section set in an unnumbered
    # document.
    own = {m.group(1) for m in HEADING_NUM_RE.finditer(FENCE_BLOCK_RE.sub("", text))}
    if not own:  # not a numbered document; nothing to resolve against
        return
    for m in SEE_SECTION_RE.finditer(text):
        if m.group(1) not in own:
            yield f"§{m.group(1)}"


def sources() -> Iterator[Path]:
    """Documents scanned as link SOURCES: the maintained layer, current by contract.

    `.html` under `docs/` is scanned too, so a `.md` and its rendered companion stay a pair to
    the gate. Third-party input and vendored trees (`build/`, `formal/lean/.lake`) are not
    ours to grade. A missing directory contributes no sources.
    """
    for name in ("CLAUDE.md", "README.md"):
        if (p := REPO / name).exists():
            yield p
    yield from sorted((REPO / "wiki").rglob("*.md"))
    for pattern in ("*.md", "*.html"):
        yield from sorted((REPO / "docs").rglob(pattern))


def scan() -> list[Violation]:
    adrs = adr_index()
    seen: set[tuple[str, str]] = set()
    found: list[Violation] = []
    for src in sources():
        rel = src.relative_to(REPO).as_posix()
        text = src.read_text(encoding="utf-8", errors="replace")
        refs = [
            *((p, Kind.GONE) for p in links(text) if not resolves(p)),
            *((r, Kind.ADR_UNKNOWN) for r in unknown_adrs(text, adrs)),
            *((r, Kind.SECTION) for r in dangling_sections(text)),
        ]
        for ref, kind in refs:
            if (rel, ref) not in seen:
                seen.add((rel, ref))
                found.append(Violation(rel, ref, kind))
    return sorted(found, key=lambda v: (v.source, v.path))


# --- baseline ---------------------------------------------------------------


def load_baseline() -> dict[tuple[str, str], str]:
    if not BASELINE.exists():
        return {}
    entries: dict[tuple[str, str], str] = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        if not (line := line.strip()) or line.startswith("#"):
            continue
        source, path, reason = (part.strip() for part in line.split("::", 2))
        entries[(source, path)] = reason
    return entries


def write_baseline(entries: dict[tuple[str, str], str]) -> None:
    header = [
        "# link-check baseline: pre-existing dead documentation links.",
        "#",
        "# Format:  <source> :: <dead path> :: <reason>",
        "#",
        "# The gate fails on any violation NOT listed here, AND on any entry here",
        "# that matches no current violation (stale: delete it). Entries are keyed",
        "# by SOURCE, so moving a document orphans every entry whose source it was;",
        "# a move commit rewrites baseline KEYS, not just the paths inside them.",
        "#",
        "# Do not hand-edit to silence a new break: a new break is fixed, never recorded.",
        "",
    ]
    body = (
        f"{source} :: {path} :: {reason}" for (source, path), reason in sorted(entries.items())
    )
    BASELINE.write_text("\n".join([*header, *body, ""]), encoding="utf-8")


def rewrite(manifest: Path) -> int:
    """Apply enumerated `old :: new :: expected_hits` citation rewrites across the sources.

    Each pair carries an expected hit count, and the whole run refuses if any count differs: a
    citation can be a glob, and a matcher that changed 6 of 8 while reporting "8 done" would
    leave the move half-made. The run is idempotent: once `docs/` has become `third-party-docs/`,
    the new path contains the old one, and the lookbehind keeps a replace from re-applying to its
    own output.
    """
    pairs: list[tuple[str, str, int]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if (line := line.strip()) and not line.startswith("#"):
            old, new, count = (p.strip() for p in line.split("::", 2))
            pairs.append((old, new, int(count)))

    targets = [(src, src.read_text(encoding="utf-8")) for src in sources()]
    # Two guards:
    #   (?<![\w/-])  a bare basename must not match inside a full path, and `docs/` must not
    #                match inside `third-party-docs/`.
    #   (?!:\d)      a bare path must not swallow a LINE-ANCHORED citation: `x.md:1035` has to
    #                be re-resolved against the moved file, which this cannot verify, so it
    #                refuses to guess.
    patterns = {old: re.compile(rf"(?<![\w/-]){re.escape(old)}(?!:\d)") for old, _, _ in pairs}

    actual = {
        old: sum(len(patterns[old].findall(text)) for _, text in targets) for old, _, _ in pairs
    }
    if mismatched := [(old, want, actual[old]) for old, _, want in pairs if actual[old] != want]:
        print("REFUSED: manifest hit counts do not match the corpus.\n", file=sys.stderr)
        for old, want, got in mismatched:
            print(f"  {old}\n      expected {want}, found {got}", file=sys.stderr)
        print("\n  Nothing was rewritten. Re-measure before moving anything.", file=sys.stderr)
        return 2

    changed = 0
    for src, text in targets:
        updated = text
        for old, new, _ in pairs:
            updated = patterns[old].sub(new, updated)
        if updated != text:
            src.write_text(updated, encoding="utf-8")
            changed += 1
    print(f"rewrote {sum(a for a in actual.values())} citation(s) across {changed} file(s)")
    return 0


def prune_stale(
    baseline: dict[tuple[str, str], str], violations: list[Violation]
) -> list[tuple[str, str]]:
    """Drop entries matching no current violation, the one edit the stale rule demands.

    It can only remove a recorded violation that no longer reproduces.
    """
    live = {v.pair for v in violations}
    stale = sorted(baseline.keys() - live)
    write_baseline({pair: reason for pair, reason in baseline.items() if pair in live})
    return stale


# --- reporting --------------------------------------------------------------


def report(new: list[Violation], stale: list[tuple[str, str]], verbose: bool) -> None:
    if new:
        head = new if verbose else new[:20]
        print(f"\n{len(new)} NEW dead documentation link(s), not in the baseline:\n")
        for v in head:
            print(f"  {v.source}\n      → {v.path}  ({v.kind})")
        if len(new) > len(head):
            print(f"  … and {len(new) - len(head)} more (--verbose)")
        print("\n  Fix the link.")
    if stale:
        print(f"\n{len(stale)} STALE baseline entr(ies): fixed, but still recorded:\n")
        for source, path in stale:
            print(f"  {source} :: {path}")
        print("\n  Delete these lines from scripts/link-check-baseline.txt.")
        print("  Left in place, the identical break can recur and be absorbed in silence.")


def main() -> int:
    args = sys.argv[1:]
    verbose = "--verbose" in args
    violations = scan()

    if "--init-baseline" in args:
        if BASELINE.exists():
            print(
                f"{BASELINE.relative_to(REPO)} already exists: "
                "--init-baseline is a one-time bootstrap; fix the link instead.",
                file=sys.stderr,
            )
            return 2
        # One pass is the fixpoint: the baseline resolves by construction (BASELINE_PATH),
        # so writing it changes no verdict.
        write_baseline({v.pair: _reason(v) for v in violations})
        print(f"wrote {BASELINE.relative_to(REPO)} with {len(violations)} entries")
        return 0

    if "--rewrite" in args:
        return rewrite(Path(args[args.index("--rewrite") + 1]))

    baseline = load_baseline()
    new = [v for v in violations if v.pair not in baseline]
    stale = sorted(baseline.keys() - {v.pair for v in violations})

    if "--prune-stale" in args:
        dropped = prune_stale(baseline, violations)
        print(
            f"pruned {len(dropped)} stale baseline entr(ies); "
            f"{len(baseline) - len(dropped)} remain"
        )
        return 0

    report(new, stale, verbose)
    cited = cited_paths()
    if declarations := stale_generated(cited):
        print(
            f"\n{len(declarations)} STALE generated-artifact declaration(s) in "
            "link_check.GENERATED:\n"
        )
        for path, why in declarations:
            print(f"  {path}: {why}")
        print("\n  Delete the entry. The table is a hole in the gate's domain, so it")
        print("  is held to the same ratchet as the baseline: it can only shrink.")
    if new or stale or declarations:
        extra = f", {len(declarations)} stale declaration(s)" if declarations else ""
        print(
            f"\nFAIL: {len(new)} new, {len(stale)} stale{extra} (baseline holds {len(baseline)})"
        )
        return 1
    held = sum(cited.get(path, 0) for path in GENERATED)
    print(
        f"docs-check: {len(violations)} known violation(s) baselined, "
        f"{held} citation(s) held by declaration, 0 new, 0 stale"
    )
    return 0


def _reason(v: Violation) -> str:
    """A uniform tag: growth is guarded by the stale rule and the bootstrap's one-time write,
    so a per-line reason would add nothing a reviewer reads."""
    return "forward-ref" if v.path.startswith("docs/") else "pre-gate-rot"


if __name__ == "__main__":
    sys.exit(main())
