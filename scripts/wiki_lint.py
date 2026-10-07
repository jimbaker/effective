"""The wiki's structural lint, made runnable.

Structure only. It checks that every link between pages resolves and that every
page is reachable, since both rot silently as pages are added. A link between
pages is a relative markdown link, `[name](other.md)`, so GitHub and an editor
follow it as written. Contradictions between
pages, a claim a newer document has superseded, and a task that is quietly done
need a reader, and that half of the lint pass stays a reading job.

A `wiki/CLAUDE.md`, when present, is the schema rather than a page, so it is
neither linted for inbound links nor read for outbound ones: it quotes the link
syntax while explaining it.

When the wiki carries a `deferred.md`, it also evaluates the revival conditions
there. An option is deferred under a condition, and the condition coming true is
exactly what nobody notices. So a condition that can be written as a command is
written as one, and a fired condition FAILS -- the entry then gets acted on, or
its condition restated, but it cannot sit. A `tasks.md` queue, when present, has
its items held to a length bound. Both pages are optional.

**The domain is reported, not assumed.** Most conditions are judgement (*"when a
reader is observed confusing them"*), and a gate that quietly evaluated the
mechanical few would report green over a page whose hard conditions are all
unchecked. So the count of unevaluated entries is printed every run, beside the
ones that were checked.

This is a sibling of `scripts/link_check.py`, which grades the *citation* forms
the maintained layer uses (repo-rooted paths, ADR numbers, `see §N` pointers).
This one grades the links that join wiki pages; link_check grades the citations
that leave the wiki.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

try:  # imported as a package member (tests, `from scripts.wiki_lint import …`)
    from scripts.link_check import destinations
except ImportError:  # run as a script: scripts/ is on sys.path, the repo root is not
    from link_check import destinations

ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "wiki"
SCHEMA = "CLAUDE"
ENTRY = {"index", "log"}

LINK = re.compile(r"\[\[([^\]]+)\]\]")
CODE = re.compile(r"`[^`]*`")
FENCED = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)


def pages() -> dict[str, set[str]]:
    """Page name to the pages it links out to, code ignored."""
    found = {}
    for path in sorted(WIKI.rglob("*.md")):
        name = path.relative_to(WIKI).with_suffix("").as_posix()
        if name == SCHEMA:
            continue
        found[name] = outbound(path)
    return found


def outbound(path: Path) -> set[str]:
    """The wiki pages the page at `path` links to, each link resolved from it."""
    names = set()
    for target in destinations(path.read_text(encoding="utf-8")):
        resolved = Path(os.path.normpath(path.parent / target))
        if resolved.suffix == ".md" and resolved.is_relative_to(WIKI):
            names.add(resolved.relative_to(WIKI).with_suffix("").as_posix())
    return names


DEFERRED = WIKI / "deferred.md"
TASKS = WIKI / "tasks.md"
ITEM_RE = re.compile(
    r"^- \*\*`(?P<slug>[a-z0-9-]+)`\*\*.*?(?=^- \*\*`|^#|\Z)", re.MULTILINE | re.DOTALL
)
"""An item runs to the next item OR to the next heading.

Stopping only at the next item would count a section heading's preamble into the item above it,
so the last item before a heading would fail a bound it had not crossed."""
ITEM_LINES = 12
"""An item is slug, state, blocker, a one-line statement and provenance, which fits in about six.

Twelve is generous against that: it admits every item that states its slug, state, blocker, claim
and provenance, and refuses the ones that tell a story instead.

**A bound with no relocation step deletes.** Before cutting to fit, check the clause is stated
somewhere else, and put it there in the same commit."""
ENTRY_RE = re.compile(r"^## (?P<name>.+?)$(?P<body>.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
CHECK_RE = re.compile(r"^\*\*Check:\*\*\s*`(?P<cmd>[^`]+)`", re.MULTILINE)


def revivals() -> tuple[list[str], int, int]:
    """Fired conditions, how many were checked, how many could not be.

    An entry opts in by carrying a `**Check:**` line holding one shell command.
    Exit 0 means the condition has come true; anything else means not yet. An
    entry with no such line is counted as unevaluated and never as passing.
    """
    if not DEFERRED.exists():
        return [], 0, 0
    fired, checked, unevaluated = [], 0, 0
    for match in ENTRY_RE.finditer(DEFERRED.read_text(encoding="utf-8")):
        if not (cmd := CHECK_RE.search(match.group("body"))):
            unevaluated += 1
            continue
        checked += 1
        done = subprocess.run(
            cmd.group("cmd"), shell=True, cwd=ROOT, capture_output=True, timeout=60
        )
        name = match.group("name").strip()
        if done.returncode == 0:
            fired.append(f"{name}  --  `{cmd.group('cmd')}`")
        elif done.returncode > 1:
            # rc 1 is an honest "not yet"; anything above it is the instrument failing --
            # a missing tool exits 127 and would otherwise read exactly like a condition
            # that has not come true.
            fired.append(f"{name}  --  CHECK BROKE (rc {done.returncode}): `{cmd.group('cmd')}`")
    return fired, checked, unevaluated


def overlong_items() -> list[str]:
    """Queue items carrying narrative that belongs in a report they cite."""
    if not TASKS.exists():
        return []
    text = TASKS.read_text(encoding="utf-8")
    return [
        f"{m.group('slug')}  --  {n} lines, over {ITEM_LINES}"
        for m in ITEM_RE.finditer(text)
        if (n := len(m.group(0).rstrip().splitlines())) > ITEM_LINES
    ]


def bracketed() -> list[str]:
    """`[[page]]` links outside code, which GitHub renders as text."""
    return [
        f"{path.relative_to(WIKI).as_posix()}: [[{target}]]"
        for path in sorted(WIKI.rglob("*.md"))
        if path.stem != SCHEMA
        for target in LINK.findall(CODE.sub("", FENCED.sub("", path.read_text(encoding="utf-8"))))
    ]


def structure(out: dict[str, set[str]]) -> list[str]:
    """Dead links, orphan pages, and bracketed links, one line each."""
    inbound = {to for tos in out.values() for to in tos}
    return [
        *(
            f"DEAD   {src} links to {to}, which is not a page"
            for src, tos in sorted(out.items())
            for to in sorted(tos)
            if to not in out
        ),
        *(f"ORPHAN {name} has no inbound link" for name in sorted(set(out) - inbound - ENTRY)),
        *(
            f"OLD    {link}: write a relative markdown link, [page](page.md)"
            for link in bracketed()
        ),
    ]


def main() -> int:
    out = pages()
    if not out:
        print("no wiki pages found; wiki/ is missing or empty")
        return 1
    print(f"{len(out)} pages, {sum(len(v) for v in out.values())} links\n")
    for name in sorted(out):
        n_in = sum(name in tos for tos in out.values())
        print(f"  {name:34} in:{n_in:3}  out:{len(out[name]):3}")
    print()
    for finding in (broken := structure(out)):
        print(finding)
    if not broken:
        print("no dead links, no orphans")

    if bloated := overlong_items():
        print()
        for item in bloated:
            print(f"LONG   {item}")
        print("  an item is slug, state, blocker, one line and provenance; narrative goes")
        print("  in a report the item cites")

    fired, checked, unevaluated = revivals()
    print(f"\nrevival conditions: {checked} checked, {unevaluated} need a reader")
    for entry in fired:
        print(f"FIRED  {entry}")
    if fired:
        print("  act on it, or restate the condition -- a fired condition may not sit")
    return 1 if broken or fired or bloated else 0


if __name__ == "__main__":
    sys.exit(main())
