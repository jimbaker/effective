"""Mermaid fence linter: the cheap check to run when you write a diagram.

A mermaid fence that fails to parse renders as a raw error box in every venue that
renders mermaid. Two of the three sequence diagrams in `docs/effective-101.md`
shipped that way and nothing caught it: the link gate checks that a document's
*paths* resolve, not that its *diagrams* render.

Deliberately a SYNTAX check. `mmdc` needs a ~500 MB container, so gating on a render
would put podman between you and a green suite; this runs in milliseconds and catches
the fatal-and-silent class. Render to PNG and look at the raster when authoring, and
run this every time.

    just lint-mermaid docs/effective-101.md   # one file
    just lint-mermaid docs/                   # a directory
    just lint-mermaid                         # defaults to docs/

Provenance decides how much to trust each rule: the two SEQUENCE rules are failures
this repo actually hit. The flowchart rule came from authoring guidance, and it
produced two false positives against the real corpus before it was right. A rule
you have not seen fire in anger needs its negative cases pinned hardest.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FENCE_RE = re.compile(r"```mermaid\n(.*?)```", re.S)
# a sequence-diagram statement carrying free text: `A->>B: text`, `Note over X: text`
MESSAGE_RE = re.compile(
    r"^\s*(?:\w+\s*-?-+>>?\s*\w+|Note (?:over|left of|right of) [\w, ]+)\s*:\s*(.*)$"
)
# mermaid's character escape, `#59;` or `#quot;`: it renders as the character it names
ENTITY_RE = re.compile(r"#\w+;")
NOTE_RE = re.compile(r"\s*Note (?:over|left of|right of) ")
# what a line after a Note may legitimately start with
STATEMENT_RE = re.compile(
    r"(\w+\s*-?-+>>?|Note |participant |actor |alt |else |end|opt |loop |par |and |"
    r"rect |activate |deactivate |autonumber|critical |break |box )"
)
# A flowchart node declaration: `id`, its shape opener run, then the label UP TO the
# first closer. Shape openers stack (`[(` cylinder, `[[` subroutine, `((` circle), and
# the label must stop at the closer — running it to end-of-line makes `A[ask] --> B{x}`
# look like a label containing `{`, which is the second false positive this rule
# produced. Both are pinned as negative cases in the tests.
NODE_RE = re.compile(r"\b\w[\w-]*([\[\(\{]+)([^\]\)\}]*)")
BRACKET = set("([{")


@dataclass(frozen=True)
class Finding:
    path: Path
    fence: int
    line: int
    rule: str
    detail: str
    fix: str


def fences(text: str) -> list[str]:
    return FENCE_RE.findall(text)


def check_block(path: Path, index: int, block: str) -> list[Finding]:
    """Every fatal-and-silent rule, applied to one fence."""
    out: list[Finding] = []
    lines = block.splitlines()
    is_sequence = "sequenceDiagram" in block

    for n, line in enumerate(lines, 1):
        if (
            is_sequence
            and (m := MESSAGE_RE.match(line))
            and re.search(r"[<>;]", ENTITY_RE.sub("", m.group(1)))
        ):
            # `;` is a statement separator and `<`/`>` are arrow-operator parts, so
            # both end the message early. A `<br/>` for a line break trips this too.
            # An entity (`#59;`) is mermaid's escape and renders the character, so
            # its own `;` does not count.
            out.append(
                Finding(
                    path,
                    index,
                    n,
                    "fatal-char-in-message",
                    f"`;` or `<`/`>` in message text: {m.group(1).strip()[:70]}",
                    "write `;` as `#59;` where the text needs one (a key), else a comma; "
                    "mermaid wraps long text on its own",
                )
            )
        if (
            is_sequence
            and NOTE_RE.match(line)
            and n < len(lines)
            and (nxt := lines[n].strip())
            and not STATEMENT_RE.match(nxt)
        ):
            # learned while FIXING the <br/> above: the obvious replacement for a
            # line break is a line break, and that is also a parse error
            out.append(
                Finding(
                    path,
                    index,
                    n,
                    "multi-line-note",
                    f"Note continues onto the next line: {nxt[:70]}",
                    "keep a Note on one line; mermaid wraps it for you",
                )
            )
        if not is_sequence and (m := NODE_RE.search(line)):
            # Only a label that is BOTH unquoted AND carries a bracket is fatal:
            # `id(Proceed (WF-B))` dies on the inner `(`, while `n2[("text")]` is a
            # quoted cylinder and parses fine. The first version of this rule flagged
            # the cylinder — a false positive found by rendering it, which is why the
            # negative case is pinned in the tests alongside the positive one.
            label = m.group(2)
            if label[:1] not in {'"', "'"} and BRACKET & set(label):
                out.append(
                    Finding(
                        path,
                        index,
                        n,
                        "unquoted-flowchart-label",
                        f"unquoted bracket inside a node label: {line.strip()[:70]}",
                        'quote the whole label: id("Proceed (WF-B)")',
                    )
                )
    return out


def check_file(path: Path) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [f for i, block in enumerate(fences(text), 1) for f in check_block(path, i, block)]


DEFAULT_ROOTS = ("docs",)


def targets(args: list[str]) -> list[Path]:
    """Explicit paths, else `docs/`, which is undated and maintained, so "current and clean"
    is its contract.
    """
    roots = [Path(a) for a in args if not a.startswith("-")] or [REPO / r for r in DEFAULT_ROOTS]
    out: list[Path] = []
    for root in roots:
        root = root if root.is_absolute() else REPO / root
        if root.is_dir():
            out.extend(sorted(root.rglob("*.md")))
        elif root.exists():
            out.append(root)
        else:
            print(f"lint-mermaid: no such path: {root}", file=sys.stderr)
            raise SystemExit(2)
    return out


def main() -> int:
    args = sys.argv[1:]
    paths = targets(args)
    findings = [f for p in paths for f in check_file(p)]
    scanned = sum(len(fences(p.read_text(encoding="utf-8", errors="replace"))) for p in paths)

    if findings:
        files = len({f.path for f in findings})
        print(f"\n{len(findings)} mermaid problem(s) in {files} file(s):\n")
        for f in findings:
            rel = f.path.relative_to(REPO) if f.path.is_relative_to(REPO) else f.path
            print(f"  {rel}  (fence #{f.fence}, line {f.line})")
            print(f"    [{f.rule}] {f.detail}")
            print(f"    fix: {f.fix}")
        print("\nA fence that does not parse renders as an error box wherever mermaid renders.")
        print("Render it to an image and look at it before trusting it.")
        return 1

    print(f"lint-mermaid: {scanned} fence(s) in {len(paths)} file(s), no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
