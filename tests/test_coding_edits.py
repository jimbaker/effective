"""The edit-scheme ladder, and the runner table that serves it.

**Roles.** The rung comparison is `adversarial`: it exists to show the two lower rungs
getting a real case WRONG, and it would be worthless if it only showed the top rung right. The
table tests are `unit`. The runner tests are `spine`: they drive the dispatch a machine actually
uses.

The case in `test_the_three_rungs_disagree_and_only_the_top_one_is_right` is the ladder's
discriminator, and it is the entire argument for paying for jedi. If it ever stops discriminating
(if text and syntactic start agreeing with semantic), the ladder has no justification and this
file should say so loudly rather than quietly passing.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from effective.coding import runners
from effective.coding.edits.semantic import SemanticError, declared_arms, references, rename
from effective.coding.edits.structural import (
    StructuralRule,
    apply_structural,
    ast_grep_bin,
    structural_matches,
)
from effective.coding.gate import static_diagnostic, static_merge_gate
from effective.coding.runners import (
    CODING_TOOLS,
    TOOLS,
    ToolError,
    UnknownTool,
    Workspace,
    run_tool,
)
from effective.machine.trampoline import SUITE_TOOL

TWO_SCOPES = """def f():
    x = 1
    return x


def g():
    x = 2
    return "x" + str(x)
"""

needs_ast_grep = pytest.mark.skipif(ast_grep_bin() is None, reason="ast-grep not installed")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(TWO_SCOPES)
    return tmp_path


# --- the ladder: the case that justifies the top rung -------------------------------------------


def test_the_three_rungs_disagree_and_only_the_top_one_is_right(project: Path):
    """Rename `x` in `f()` only. `g()` has its own `x`, and there is a string `"x"`.

    This is the measurement the whole ladder rests on, so it is asserted as a DISAGREEMENT rather
    than as three separate facts: the value of rung 3 is precisely that rungs 1 and 2 are wrong
    here, and a test that only checked jedi would still pass on a day the others caught up."""
    source = (project / "mod.py").read_text()

    # Rung 1 — bytes. Renames both scopes AND the string.
    textual = re.sub(r"\bx\b", "y", source)
    assert textual.count("y") > 4
    assert '"x"' not in textual, "rung 1 rewrote the string literal — that is the defect"

    # Rung 2 — the parse. Sees every `x` NODE, in both scopes; the string is not a node, so it
    # survives, but the two scopes are still conflated.
    from ast_grep_py import SgRoot

    nodes = SgRoot(source, "python").root().find_all(pattern="x")
    assert len(nodes) == 4, "rung 2 should see four `x` identifier nodes across both scopes"

    # Rung 3 — bindings. Two references, both inside `f`.
    refs = references(project, "mod.py", line=2, column=4)
    assert len(refs) == 2
    assert {r.line for r in refs} == {2, 3}, "rung 3 must resolve `f`'s binding only"


def test_a_semantic_rename_leaves_the_other_scope_and_the_string_alone(project: Path):
    edit = rename(project, "mod.py", line=2, column=4, new_name="y")
    assert edit.diff, "a rename that changes nothing is not a rename"
    assert "-    x = 1" in edit.diff
    assert "+    y = 1" in edit.diff
    # The proof: `g`'s own binding and the string never appear as removals.
    assert "-    x = 2" not in edit.diff
    assert '-    return "x"' not in edit.diff


def test_a_rename_returns_a_diff_and_writes_nothing(project: Path):
    """The intent is recordable and applying it is a separate, explicit act — which is what keeps
    a semantic edit inside the determinism boundary."""
    before = (project / "mod.py").read_text()
    rename(project, "mod.py", line=2, column=4, new_name="y")
    assert (project / "mod.py").read_text() == before


def test_declared_arms_resolves_a_PEP695_union(tmp_path: Path):
    """FINALIZE's analysis: knowing a union's DECLARED arms, which is a type fact no lower rung
    can reach."""
    (tmp_path / "u.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\nclass A: pass\n"
        "@dataclass(frozen=True)\nclass B: pass\n"
        "type U = A | B\n"
    )
    assert set(declared_arms(tmp_path, "u.py", "U")) == {"A", "B"}


def test_semantic_failures_are_raised_not_swallowed(project: Path):
    """A refusal and an empty answer are different facts; collapsing them is how a rename
    silently does nothing."""
    with pytest.raises(SemanticError):
        rename(project, "mod.py", line=2, column=4, new_name="not an identifier")
    with pytest.raises(SemanticError):
        declared_arms(project, "mod.py", "NoSuchAlias")
    with pytest.raises(SemanticError):
        references(project, "nope.py", line=1, column=0)


# --- rung 2's oracle -----------------------------------------------------------------------------


@needs_ast_grep
def test_the_match_count_is_the_oracle_not_the_exit_code():
    """`ast-grep run` exits 1 on no-match (like grep) and 0 on a MALFORMED pattern with only a
    stderr warning — so a caller trusting the exit code reads a typo as success. The binding's
    count is what the channel turns into a `Repair`."""
    src = "foo(1)\nfoo(2)\n"
    assert structural_matches(src, "foo($A)") == 2
    assert structural_matches(src, "bar($A)") == 0
    assert structural_matches(src, "((((") == 0  # malformed, and it must not raise

    sg = ast_grep_bin()
    assert sg is not None
    proc = subprocess.run(
        [sg, "run", "--lang", "python", "-p", "((((", "--json=compact"],
        input=src, capture_output=True, text=True, timeout=30,
    )  # fmt: skip
    assert proc.returncode == 0, "a malformed pattern still exits 0 — hence the oracle"


@needs_ast_grep
def test_apply_structural_rewrites_and_reports_its_reach():
    content, total, error = apply_structural(
        "foo(1)\nfoo(2)\n", [StructuralRule(pattern="foo($A)", rewrite="bar($A)")], "m.py"
    )
    assert error is None
    assert total == 2
    assert content is not None
    assert "bar(1)" in content
    assert "foo(" not in content


def test_a_missing_ast_grep_refuses_the_rewrite(monkeypatch):
    """The rung that already has the right polarity, asserted rather than assumed.

    `apply_structural` returns `"ast-grep unavailable"` and the tool raises. No other test reaches
    this branch: it is unreachable on any box that has ast-grep, which is every box that runs the
    suite."""
    monkeypatch.setattr("effective.coding.edits.structural.ast_grep_bin", lambda: None)
    content, total, error = apply_structural(
        "foo(1)\n", [StructuralRule(pattern="foo($A)", rewrite="bar($A)")], "m.py"
    )
    assert content is None
    assert total == 0
    assert error == "ast-grep unavailable"


@needs_ast_grep
def test_a_missing_ruff_refuses_the_rewrite(monkeypatch):
    """The same polarity question, one function over.

    `apply_structural` runs `ruff --fix` after the rewrite. Skipping that step with ruff absent
    would return the unformatted bytes, so a replay on a different box would re-derive different
    output; a `ToolError` keeps the delivered bytes the same on every box."""
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda n: None if n == "ruff" else real(n))
    _content, _total, error = apply_structural(
        "foo(1)\n", [StructuralRule(pattern="foo($A)", rewrite="bar($A)")], "m.py"
    )
    assert error == "ruff unavailable"


# --- the gate -------------------------------------------------------------------------------


def test_the_fidelity_floor_catches_what_every_other_gate_misses():
    """A control byte passes ruff, ty, effective.lint and import alike. Normal Unicode does not
    trip it — an em-dash is printable, so this is not a false-positive machine."""
    assert static_diagnostic("x = 1  # \n", "m.py") is not None
    assert static_diagnostic("x = 1  # an em-dash — is fine\n", "m.py") is None


def test_the_gate_refuses_unparseable_and_unlintable_content():
    assert "syntax" in (static_diagnostic("def (:\n", "m.py") or "")
    assert static_diagnostic("import os\n", "m.py") is not None  # unused import -> ruff
    assert static_merge_gate({"a.py": "x = 1\n", "b.py": "def (:\n"}) is not None
    assert static_merge_gate({"a.py": "x = 1\n"}) is None


def test_the_gate_leaves_non_python_alone_except_for_fidelity():
    assert static_diagnostic("not python at all", "notes.md") is None
    assert static_diagnostic("bad  byte", "notes.md") is not None


def test_the_gate_determinism_lints_a_file_the_machine_WRITES():
    """`static_diagnostic` dispatches the determinism lint on `is_workflow_role(name)` — membership
    in a hand-curated 21-entry tuple — and a file the machine writes is never in it. So the
    machine's own closing predicate is blind to the boundary it exists to protect: **identical
    bytes are refused under a role-set name and accepted under a new one.**

    The remedy is the operational definition `--role-coverage` already owns — *does this content
    yield an effect op?* — so the gate stops asking who the author is and asks what the file does.
    """
    body = 'from effective import step\n\n\ndef w():\n    yield step("a", None)\n'
    named = static_diagnostic(body, "src/effective/react.py")  # in the curated tuple
    assert "require-yield-from" in (named or "")
    written = static_diagnostic(body, "src/effective/newly_written_by_the_machine.py")
    assert "require-yield-from" in (written or "")  # same bytes, a name nobody curated


# --- the runner table: the point of the promotion --------------------------------------------


def test_every_tool_in_the_table_has_a_runner():
    """The pin for the rule: no entry may be a name whose only implementation is a test or a
    script. Callable and non-None is the weak half; the tests below drive the real bodies."""
    assert TOOLS, "an empty table serves nothing"
    for name, runner in TOOLS.items():
        assert callable(runner), name
        assert runner.__module__ == "effective.coding.runners", (
            f"{name} is served from {runner.__module__} — the table must own its bodies"
        )


def test_the_predicate_the_machine_yields_is_a_tool_the_table_serves():
    """One mint, two consumers. The postamble yields `SUITE_TOOL` and the table serves it; if the
    two drifted by a typo the machine would call a tool nothing implements, and the failure would
    arrive at the end of a run rather than at import."""
    assert SUITE_TOOL in TOOLS


def test_an_unknown_tool_raises_rather_than_defaulting():
    """A silently-ignored tool call is how an agent appears to work while doing nothing."""
    with pytest.raises(UnknownTool):
        run_tool(Workspace(), "no_such_tool", {})


def test_the_workspace_tools_read_and_write_through_the_tree():
    ws = Workspace(tree={"m.py": "x = 1\n"})
    assert run_tool(ws, "read_file", {"path": "m.py"}) == "x = 1\n"
    assert run_tool(ws, "list_dir", {}) == ["m.py"]
    assert run_tool(ws, "check", {}) == "clean"
    written = run_tool(ws, "write_file", {"path": "n.py", "content": "y = 2\n"})
    assert written == {"m.py": "x = 1\n", "n.py": "y = 2\n"}


EDITS = {
    "write_file": {"path": "n.py", "content": "y = 2\n"},
    "structural_apply": {"path": "m.py", "rules": [{"pattern": "1", "rewrite": "2"}]},
}
"""An accepted call of each tool that returns a tree."""


@needs_ast_grep
@pytest.mark.parametrize("tool", list(EDITS))
def test_an_edit_leaves_the_tree_it_was_handed_unchanged(tool):
    """What lets a domain keep nothing between calls: a tool hands back the tree it makes, and the
    tree it was handed stays as the record describes it."""
    handed = {"m.py": "x = 1\n"}
    made = run_tool(Workspace(tree=handed), tool, EDITS[tool])
    assert made != handed
    assert handed == {"m.py": "x = 1\n"}


def test_a_write_returns_the_tree_it_checked(monkeypatch):
    """The tree handed in changes while the static gate runs, as a dict shared with its caller
    can; what comes back is still the tree that was checked."""
    handed = {"m.py": "x = 1\n"}
    checked = runners.static_diagnostic

    def gate(content: str, path: str) -> str | None:
        handed["../outside.py"] = "X = 1\n"
        return checked(content, path)

    monkeypatch.setattr(runners, "static_diagnostic", gate)
    made = run_tool(Workspace(tree=handed), "write_file", EDITS["write_file"])
    assert made == {"m.py": "x = 1\n", "n.py": "y = 2\n"}


@pytest.mark.parametrize("tool", ["run_suite", "list_dir", "check"])
def test_a_tool_that_takes_nothing_refuses_what_a_model_names(tool):
    """A tree a model names for `run_suite` would otherwise be replaced by the one `coding_bind`
    binds, and the model told about a measurement of a tree it did not name."""
    with pytest.raises(ValidationError):
        CODING_TOOLS[tool].args.model_validate({"tree": {"m.py": ""}})


def test_a_write_that_would_not_merge_is_refused():
    """The gate is wired into the runner, not merely available beside it."""
    ws = Workspace(tree={})
    with pytest.raises(ToolError, match="syntax"):
        run_tool(ws, "write_file", {"path": "m.py", "content": "def (:\n"})


MERGEABLE = """def foo(n: int) -> int:
    return n


def bar(n: int) -> int:
    return n + 1


print(foo(1))
print(foo(2))
"""


@needs_ast_grep
def test_structural_apply_is_served_by_a_production_runner():
    """The gap this phase closes. The incumbent had this tool NAME with implementations only in
    test and campaign code.

    The fixture defines both names on purpose: an earlier version rewrote calls to an undefined
    `bar`, and the runner refused it with `ruff F821`. That was the gate working — a rewrite that
    does not merge is not a rewrite — so the test was wrong, not the runner."""
    ws = Workspace(tree={"m.py": MERGEABLE})
    out = run_tool(
        ws,
        "structural_apply",
        {"path": "m.py", "rules": [{"pattern": "foo($A)", "rewrite": "bar($A)"}]},
    )
    # The RESULTING TREE, not a message: the workflow may only learn what changed from a recorded
    # op result, or a replay (where no tool runs) re-derives different state.
    assert list(out) == ["m.py"]
    assert "print(bar(1))" in out["m.py"]
    assert "print(foo(" not in out["m.py"]


@needs_ast_grep
def test_a_rewrite_that_would_not_merge_is_refused_and_lands_nothing():
    """The gate is wired into the structural runner too, not only into `write_file`. Rewriting
    calls to a name nothing defines is the realistic version of this failure."""
    ws = Workspace(tree={"m.py": MERGEABLE})
    with pytest.raises(ToolError, match="does not merge"):
        run_tool(
            ws,
            "structural_apply",
            {"path": "m.py", "rules": [{"pattern": "foo($A)", "rewrite": "undefined_name($A)"}]},
        )


@needs_ast_grep
def test_a_pattern_that_matches_nothing_is_an_error_not_a_silent_no_op():
    ws = Workspace(tree={"m.py": "foo(1)\n"})
    with pytest.raises(ToolError, match="0 sites"):
        run_tool(
            ws,
            "structural_apply",
            {"path": "m.py", "rules": [{"pattern": "nope($A)", "rewrite": "x($A)"}]},
        )


def test_the_semantic_tools_need_a_project_and_say_so(project: Path):
    ws = Workspace(tree={}, project_root=None)
    with pytest.raises(ToolError, match="project_root"):
        run_tool(
            ws, "semantic_rename", {"path": "mod.py", "line": 2, "column": 4, "new_name": "y"}
        )

    ws = Workspace(tree={}, project_root=str(project))
    diff = run_tool(
        ws, "semantic_rename", {"path": "mod.py", "line": 2, "column": 4, "new_name": "y"}
    )
    assert "+    y = 1" in diff


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
@pytest.mark.parametrize("missing", ["ruff"])
def test_a_missing_tool_refuses_rather_than_passing(monkeypatch, missing):
    """A verdict that depends on `PATH` is not a verdict.

    `static_diagnostic` is documented as *"a pure function of `(content, name)`"* — the sentence
    the delivery set relies on, since a durable replay re-derives a refusal rather than recording
    it. `shutil.which(...) or return None` breaks that in the worst direction: the same content is
    CLEAN on a box without the tool and refused on one with it, and clean is the answer that
    commits.

    The neighboring rung has the right polarity (`apply_structural` returns
    `"ast-grep unavailable"` and the tool raises), so this asserts the two agree rather than
    inventing a rule.

    Without this test the suite never reaches this path: every box that runs it has ruff
    installed."""
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda n: None if n == missing else real(n))
    # Content ruff rejects (an unused import) — clean to `compile`, so only ruff can refuse it.
    assert static_diagnostic("import os\n", "m.py") is not None


def test_the_gate_uses_the_repo_config_not_ruff_defaults():
    """A gate on ruff's defaults would pass content the repo rejects — the opposite of its job.
    Line length 99 is the repo's, not ruff's 88."""
    ninety = "x = '" + "a" * 80 + "'\n"
    assert static_diagnostic(ninety, "m.py") is None, "88 < len < 99 must pass under repo config"


def test_the_gate_imports_where_no_repo_config_exists(monkeypatch):
    """A config resolved at MODULE SCOPE would raise, so `import effective.coding.gate` would
    fail outright on any box that is not a source checkout, a wheel in site-packages being the
    ordinary one. It is a packaging property decided by an import side effect, and unreachable
    from inside the repo without this simulation.

    Simulated by pointing the search at a tree with no `pyproject.toml` above it rather than by
    building a wheel: the search walks `Path(__file__).resolve().parents`, so the file's location
    is the whole input."""
    from effective.coding import gate

    monkeypatch.setattr(gate, "__file__", "/nonexistent/effective/coding/gate.py")
    gate.find_ruff_config.cache_clear()
    try:
        assert gate.find_ruff_config() is None
    finally:
        # The cache is process-wide, so a stale None would leak into every later test.
        monkeypatch.undo()
        gate.find_ruff_config.cache_clear()


def test_a_missing_repo_config_refuses_rather_than_using_ruff_defaults(monkeypatch):
    """The polarity, which is the half that matters and the half the two callers disagree on.

    `ruff_diagnostic` JUDGES, so absence must not read as clean — ruff's defaults are not this
    repo's rules, and content vetted against them would pass here and fail `just lint`. Same
    reasoning as the missing-`ruff` refusal above, one rung along.

    `apply_structural` deliberately does the OPPOSITE with the same `None`, because it tidies a
    rewrite rather than judging one; that difference is now a choice at the call site instead of
    two hand-written searches that happened to differ."""
    from effective.coding import gate

    monkeypatch.setattr(gate, "find_ruff_config", lambda: None)
    verdict = gate.ruff_diagnostic("x = 1\n", "m.py")
    assert verdict is not None, "a missing config must REFUSE, never read as clean"
    assert "never vetted" in verdict, verdict
