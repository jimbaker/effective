"""Rung 2 — the syntactic editor: the model emits an ast-grep pattern, Python does the rest.

This module and `effective.coding.runners` together are the production runner for the
`structural_apply` tool name.

**The decomposition is the point.** The model's paid, nondeterministic surface shrinks to one
semantic step (propose a `$METAVAR` pattern and its rewrite); locate, rewrite, format and validate
are deterministic Python. Two failure modes disappear with it: there is no verbatim anchor, so the
escaping thrash of a text edit is gone, and `ruff --fix` handles placement, so the model never has
to sort imports.

Sweet spot: uniform structural rewrites — `foo($$$A)` to `bar($$$A)`, wrap-in-call, add a
decorator. Not the sweet spot: list manipulation, where variadic append hits trailing-comma
fragility; that stays textual, and saying so is cheaper than discovering it.

**`ruff --fix` is allowed here and refused in `gate.py`, which is not a contradiction.** The gate
must not auto-fix, because the *model* emitted the code and a fix would launder work it did not
do. Here the model emitted a *pattern*; the code is generated deterministically, so formatting it
is part of the derivation rather than a favour to the model.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from ast_grep_py import SgRoot
from pydantic import BaseModel

from effective.coding.gate import find_ruff_config


class StructuralRule(BaseModel):
    """One ast-grep pattern and its rewrite, both code-shaped `$METAVAR` templates.

    A single `$VAR` binds one node; `$$$VAR` binds a variadic run and re-expands in the
    rewrite. Patterns rather than YAML on purpose: they look like code, which is more
    in-distribution for a weak model than a rule schema."""

    pattern: str
    rewrite: str


class StructuralEdit(BaseModel):
    """The WHOLE model output for this rung — a path and its rules. Recorded as the `AskLLM`
    result, so a replay re-derives the content from the intent rather than trusting a stored
    diff."""

    path: str
    rules: list[StructuralRule]


def ast_grep_bin() -> str | None:
    """The CLI, if present. `ast-grep` and `sg` are the same binary under two names."""
    return shutil.which("ast-grep") or shutil.which("sg")


def structural_matches(content: str, pattern: str) -> int:
    """How many sites `pattern` hits — **the authoritative oracle**, and richer than a lint line
    because it answers "your rule hit N sites" rather than "something went wrong".

    It has to be the oracle, because `ast-grep run`'s exit code is not a usable error signal: a
    no-match exits 1 like grep, and a MALFORMED pattern exits 0 with only a stderr warning. So a
    caller that trusted the exit code would read a typo'd pattern as success. A malformed pattern
    counts 0 here, which the channel turns into a `Repair`.

    The binding is imported at module level: it is a declared runtime dependency, so a lazy
    import would only hide the one failure that should be loud (it is missing). The `except` is
    therefore narrowed to what a malformed PATTERN raises."""
    try:
        return len(SgRoot(content, "python").root().find_all(pattern=pattern))
    except ValueError, TypeError, RuntimeError:
        return 0


def apply_structural(
    content: str, rules: list[StructuralRule], name: str
) -> tuple[str | None, int, str | None]:
    """Apply every rule, then format. Returns `(new_content, total_matches, error)`.

    A pure function of its inputs **given both tools are present**: ast-grep and ruff are
    deterministic, and either one missing REFUSES rather than degrading, so a replay re-running it
    gets byte-identical output. Skipping the format step when ruff is absent would return
    different bytes for the same inputs.

    Subprocesses are fine: this is handler-side, past
    the yield boundary, where the determinism rule constrains the workflow rather than the tool."""
    sg = ast_grep_bin()
    if sg is None:
        return (None, 0, "ast-grep unavailable")
    total = sum(structural_matches(content, rule.pattern) for rule in rules)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / Path(name).name
        f.write_text(content)
        for rule in rules:
            subprocess.run(
                [sg, "run", "--lang", "python",
                 "-p", rule.pattern, "-r", rule.rewrite, "-U", str(f)],
                capture_output=True,
                text=True,
                timeout=30,
            )  # fmt: skip
        ruff = shutil.which("ruff")
        if ruff is None:
            return (None, 0, "ruff unavailable")
        if name.endswith(".py"):
            # Ruff's defaults if the repo config is not found, and that is deliberate: this
            # TIDIES a rewrite, it does not judge one. `gate.ruff_diagnostic` refuses in the same
            # situation because a verdict reached without the repo's rules is not a verdict.
            cfg = find_ruff_config()
            subprocess.run(
                [ruff, "check", "--fix", "--quiet",
                 *(("--config", str(cfg)) if cfg else ()), str(f)],
                capture_output=True,
                timeout=30,
            )  # fmt: skip
        return (f.read_text(), total, None)
