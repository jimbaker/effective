"""Every `just …` command cited in the maintained layer must actually run.

A document that tells a reader to run a command is making a checkable claim, and this repo holds
a doc that fails a grep to be a correctness bug rather than untidiness. The characteristic
failure is a citation such as

    just lint --sql-templates      just lint --working-notes

which names a real `effective.lint` mode, so the prose reads correct to anyone who knows the flag
exists. `just` disagrees: `lint` is a recipe that takes no parameters, so the flag is not an
argument, it is an unknown recipe name, and the command exits non-zero.

**The grammar this scans, stated because a gate's grammar IS its domain**: a backticked
`` `just X …` `` span, which is how every citation in the tree is written, where `X` is a recipe
declaring **no parameters** and the citation passes it arguments. `lint` *is* a recipe, so "does
this recipe exist" answers yes for such a citation; the flag is what makes it fail.
Parameterized recipes (`docs-check *ARGS`) legitimately take arguments and are left alone.

**It deliberately does NOT check that the recipe exists.** That half has a false-positive class
this one does not: a design document may cite a recipe it is *proposing*, as the channel-manifest
design does with `just gen-channel-stubs`, and flagging it would be flagging correct work, which
is how a gate gets disabled. The existence half can land with a baseline the way `link_check`
handles pre-existing rot.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCANNED = ("docs", "wiki")
CITATION = re.compile(r"`just ([a-z0-9][a-z0-9-]*)((?: [^`]*)?)`")


def _recipes() -> dict[str, bool]:
    """Recipe name → whether it declares parameters, straight from `just`'s own dump."""
    dump = subprocess.run(
        ["just", "--dump", "--dump-format", "json"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        name: bool(body.get("parameters"))
        for name, body in json.loads(dump.stdout)["recipes"].items()
    }


def _citations() -> list[tuple[Path, int, str, str]]:
    out = []
    for root in _SCANNED:
        for path in sorted((_ROOT / root).rglob("*.md")):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                for recipe, args in CITATION.findall(line):
                    out.append((path.relative_to(_ROOT), lineno, recipe, args.strip()))
    return out


def test_no_cited_just_command_passes_a_flag_to_a_parameterless_recipe():
    recipes = _recipes()
    broken = [
        f"{p}:{n}: `just {r} {a}`".rstrip()
        for p, n, r, a in _citations()
        if r in recipes and a.startswith("-") and not recipes[r]
    ]
    assert broken == [], (
        "cited `just` commands that do not run — a recipe that takes no parameters cannot "
        "accept a flag, so this is an unknown-recipe error at the reader's shell:\n  "
        + "\n  ".join(broken)
    )


def test_the_scan_reaches_its_roots_and_sees_the_shape_it_checks():
    """Anti-vacuity, asserting COVERAGE rather than a count — a count would fail the day the
    corpus legitimately shrinks. The flag-passing shape is pinned by the mutation cases below,
    since the corpus need not contain one."""
    cited = _citations()
    assert {p.parts[0] for _, _, _, p in ((0, 0, 0, c[0]) for c in cited)} >= {"docs"}


@pytest.mark.parametrize(
    ("citation", "why"),
    [
        (
            "`just lint --sql-templates`",
            "a flag on a parameterless recipe",
        ),
        ("`just lint --working-notes`", "the same shape, a different flag"),
    ],
)
def test_the_gate_would_catch_the_forms_it_claims_to(citation, why):
    """Mutation, inline: feed the checker each broken form and confirm it classifies it."""
    recipes = _recipes()
    recipe, args = CITATION.findall(citation)[0]
    assert recipe in recipes, why
    assert args.strip().startswith("-"), why
    assert not recipes[recipe], why


def test_a_legitimate_parameterised_call_is_not_flagged():
    """`docs-check *ARGS` takes arguments; flagging it would be the expensive failure — a gate
    that flags correct work gets disabled."""
    recipes = _recipes()
    assert recipes["docs-check"], (
        "docs-check must still be parameterised for this to mean anything"
    )
    recipe, args = CITATION.findall("`just docs-check --verbose`")[0]
    assert not (args.strip().startswith("-") and not recipes[recipe])
