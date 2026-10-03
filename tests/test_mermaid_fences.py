"""Mermaid fences in the prose trees must actually parse.

Two of the three sequence diagrams in `docs/effective-101.md` shipped broken and
nothing caught it — the link gate checks that a document's *paths* resolve, not that
its *diagrams* render. A fence that fails to parse renders as a raw error box in
every venue that renders mermaid.

The rules live in `scripts/lint_mermaid.py`, not here, because they are equally
useful on a report the moment it is written (`just lint-mermaid <path>`) and a rule
enforced in one place but merely *described* in the other drifts. This module gates
both trees; the recipe is the on-demand check for whatever you just wrote.

**The negative cases below are the load-bearing tests.** The flowchart rule produced
two false positives against the real corpus before it was right, and a linter with
false positives gets disabled — which costs more than the rule ever bought. Each
false positive is pinned here, and each is render-verified: mermaid itself was run on
all four fixtures, and the linter's verdict matches the renderer's on every one.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.lint_mermaid import check_block, check_file, fences  # noqa: E402

SCANNED = sorted(p for root in ("docs", "wiki") for p in (_ROOT / root).rglob("*.md"))


def ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(_ROOT)) for p in paths]


@pytest.mark.parametrize("path", SCANNED, ids=ids(SCANNED))
def test_every_mermaid_fence_parses(path):
    """`;` / `<`/`>` in message text, a Note wrapped onto a second line, an unquoted
    paren in a flowchart label — each is fatal, and each has bitten in this repo."""
    findings = check_file(path)
    assert not findings, "\n".join(
        f"{f.path.name} fence #{f.fence} line {f.line}: [{f.rule}] {f.detail} — {f.fix}"
        for f in findings
    )


# --- the rule's boundary, both directions, all render-verified ----------------

FLAGGED = [
    pytest.param("graph LR\n  a(Proceed (WF-B))\n", id="unquoted-paren-in-label"),
    pytest.param("sequenceDiagram\n  A->>B: task suspends; survives crash\n", id="semicolon"),
    pytest.param("sequenceDiagram\n  A->>B: gone<br/>here\n", id="br-tag"),
    pytest.param(
        "sequenceDiagram\n  Note over A: first line\n  second line\n", id="multi-line-note"
    ),
]

CLEAN = [
    # BOTH of these were false positives of earlier versions of the flowchart rule,
    # and BOTH render fine — confirmed by running mermaid on them.
    pytest.param('graph LR\n  n2[("ledger;step x100")]\n', id="quoted-cylinder-label"),
    pytest.param("graph LR\n  A[ask] --> B{done?}\n", id="two-nodes-on-one-line"),
    pytest.param(
        "sequenceDiagram\n  A->>B: task suspends, survives crash\n", id="comma-not-semicolon"
    ),
    pytest.param(
        'sequenceDiagram\n  K-->>S: "gather:0,1#59;rec:0/step#59;leaf"\n', id="escaped-semicolon"
    ),
    pytest.param(
        "sequenceDiagram\n  Note over A: one long line mermaid wraps\n  A->>B: next\n",
        id="single-line-note",
    ),
]


@pytest.mark.parametrize("block", FLAGGED)
def test_a_fatal_fence_is_flagged(block):
    assert check_block(Path("x.md"), 1, block), "mermaid rejects this; the linter must too"


@pytest.mark.parametrize("block", CLEAN)
def test_a_valid_fence_is_not_flagged(block):
    """False positives are the expensive failure: a linter that cries wolf gets
    disabled, and then it catches nothing at all."""
    assert not check_block(Path("x.md"), 1, block), (
        f"mermaid renders this fine; flagging it is a false positive:\n{block}"
    )


def test_the_scan_is_not_vacuous():
    """Anti-vacuity: everything above passes trivially if no fences are found."""
    blocks = [b for p in SCANNED for b in fences(p.read_text(encoding="utf-8", errors="replace"))]
    assert len(blocks) >= 8, f"expected the docs/ + wiki/ fences, found {len(blocks)}"
    assert any("sequenceDiagram" in b for b in blocks), "no sequence fences — rules vacuous"
    assert any("graph " in b or "flowchart " in b for b in blocks), "no flowchart fences"
