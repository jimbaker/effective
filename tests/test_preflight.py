"""The gate's toolchain preflight, proven on the drifts it exists to catch.

ROLE: adversarial. Every test here breaks exactly one thing and asserts the right message. A
pass means only that the attack failed, which is evidence only because each attack is one that
has happened: the interpreter moving under a green gate, `uvx ty` resolving a newer release
minutes after a clean run, and a tdom fix making the well-formed case pass while silently
degrading the malformed one.

Two more pin ways the script itself can be vacuous. Scoping the agreement check to `check`'s
closure lets `lint` and `typecheck` disagree with a green preflight. `uvx ty@0.0.999999 --version`
fails with an error that QUOTES the version it could not find, so a substring test passes on the
one input it exists to catch.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import preflight as pf  # noqa: E402

pytestmark = pytest.mark.adversarial


CLEAN = """\
lint:
    uv run ruff check src
    uvx ty@0.0.73 check

test:
    uv run pytest -q

typecheck:
    uvx ty@0.0.73 check

check: lint test
"""
"""A justfile in miniature: a gate whose closure is locked except one pinned `uvx`, plus a
standalone recipe outside the closure that pins the same tool. That last recipe is not decoration
— it is the whole of what the agreement domain is for."""


def domains(text: str) -> tuple[list[str], dict[str, list[tuple[str, str]]]]:
    """(what must be pinned and is not, where each gate tool is pinned) — the script's two."""
    recipes = pf.parse_recipes(text)
    problems, used = pf.ungoverned_tools(recipes)
    return problems, pf.pin_sites(recipes, used)


# --- the derivation: does the domain reach what it claims to? -----------------------------------


def test_the_closure_is_the_gates_dependencies_and_typecheck_is_OUTSIDE_it():
    """Vacuity check FIRST — every assertion below holds of an empty domain too.

    And the absent name is the load-bearing one: `typecheck` pins the same tool and is NOT
    reachable from `check`, which is precisely why agreement cannot be scoped to this list."""
    assert pf.closure(pf.parse_recipes(CLEAN), "check") == ["check", "lint", "test"]


def test_a_locked_command_is_not_a_finding():
    """`uv run` is governed by `uv.lock`, which IS the record. A preflight that flagged it would
    fire on every line in the gate and teach people to ignore it."""
    problems, _ = domains(CLEAN)
    assert problems == []


def test_the_real_justfile_derives_a_non_empty_domain():
    """The live gate, not a fixture. If this ever reads zero tools the script has gone blind and
    every green preflight above it means nothing."""
    recipes = pf.parse_recipes(pf.JUSTFILE.read_text())
    assert pf.GATE in recipes
    _, used = pf.ungoverned_tools(recipes)
    assert used, "no ungoverned tool found in the gate — the classifier stopped seeing `uvx`"


# --- one mutation per rule ----------------------------------------------------------------------


def test_an_ungoverned_tool_with_no_version_is_refused():
    """The rule that keeps the domain from going one entry short: a new runner in the gate is a
    finding by default, whether or not anyone remembered to list it."""
    problems, _ = domains(CLEAN.replace("    uv run pytest -q", "    npx some-linter ."))
    assert any("npx" in p and "pins no version" in p for p in problems)


def test_agreement_is_checked_ACROSS_the_file_not_only_the_gate():
    """`typecheck` sits outside `check`'s closure and is the recipe a person runs by hand. Scoping
    agreement to the closure would let the two spellings disagree about what clean means with a
    green preflight."""
    drifted = CLEAN.replace("typecheck:\n    uvx ty@0.0.73", "typecheck:\n    uvx ty@0.0.72")
    problems, _ = pf.check_pins(domains(drifted)[1])
    assert any("two versions" in p for p in problems)


def test_a_tool_pinned_outside_the_gate_ALONE_is_not_conscripted():
    """The agreement domain is gate tools at every site — not every `uvx` anywhere. A recipe
    nobody's gate reaches may pin what it likes."""
    _, sites = domains(CLEAN + "\nreport:\n    uvx cowsay@1.0 moo\n")
    assert set(sites) == {"ty"}


def test_a_pin_that_does_not_resolve_is_a_claim_rather_than_a_pin(monkeypatch):
    """`resolve` returning `None` — what a failed run must look like."""
    monkeypatch.setattr(pf, "resolve", lambda tool, version: None)
    problems, verified = pf.check_pins({"ty": [("0.0.73", "lint")]})
    assert verified == []
    assert any("could not resolve" in p for p in problems)


def test_an_error_message_quoting_the_version_is_not_a_resolution(monkeypatch):
    """THE DEFECT THE BATTERY FOUND IN THIS SCRIPT. `uvx ty@0.0.999999 --version` exits non-zero
    with *"Distribution not found … ty==0.0.999999"*. A substring test against combined output
    passes on exactly the input it exists to catch, so `resolve` reads stdout of a SUCCESSFUL run
    and nothing else — and the version match is anchored rather than substring."""
    monkeypatch.setattr(
        pf, "resolve", lambda tool, version: f"Distribution not found: ty=={version}"
    )
    problems, verified = pf.check_pins({"ty": [("0.0.999999", "lint")]})
    assert verified == []
    assert any("does not take" in p for p in problems)


def test_resolve_reads_a_SUCCESSFUL_run_only(monkeypatch):
    """The other half of the same defect, one layer down. `check_pins`' anchored match is what
    catches a quoted version in a message; this is what stops the message reaching it at all."""
    import subprocess

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "error: Distribution not found: ty==0.0.999999"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
    assert pf.resolve("ty", "0.0.999999") is None


def test_a_matching_version_verifies():
    """The green arm. Without it the four reds above could all be one broken function."""
    problems, verified = pf.check_pins({"ty": [("0.0.73", "lint")]})
    assert problems == []
    assert verified == [("ty", "0.0.73")]


# --- the two live probes ------------------------------------------------------------------------


def test_the_running_interpreter_is_the_pinned_one():
    """Not a fixture — the actual claim. `.python-version` exists precisely because nothing else
    verified it, and a suite that passes on an unpinned interpreter says less than it looks."""
    assert pf.check_interpreter() == []


def test_the_tdom_patch_is_live_on_this_interpreter():
    """Both halves, because the wrong fix passes the well-formed one. See infra/tdom/PIN.txt."""
    assert pf.check_tdom_patch() == []
