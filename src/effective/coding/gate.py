"""The static merge gate: would this content survive `just check`'s static half?

It is **handler-side**: a subprocess is fine here, because the determinism boundary constrains the
*workflow*, not the tools a handler runs past the yield. And it is a pure function of `(content,
name)` **given the toolchain is present** — ruff is deterministic, and a missing one REFUSES rather
than passing, so absence cannot masquerade as a clean verdict. That qualifier is what lets a
replay re-derive the same answer. The refusal branch is unreachable on any box that has ruff, so
a test reaches it by hiding ruff from `shutil.which`.

**The fidelity floor runs first and for every file, not just Python.** A non-printable character
passes ruff, `ty`, `effective.lint` and `import` alike; a real nano run silently swapped a
docstring em-dash for U+007F and every static gate accepted it. Normal Unicode is printable, so an
em-dash costs nothing — the check is for control bytes, and corruption in any source is corruption.

**Scope, stated because a gate's reach is easy to overclaim.** This is the STATIC half of
`just check` — parse, `effective.lint`'s role-dispatched rules, and ruff without `--fix`. It is not
`ty`, not the test suite, and not "the module imports". The narrowing matters: `effective.lint`
itself records that the code-agent gate runs 3 of the ~20 invocations in the recipe, so *"the gate
equals `just lint` by construction"* was measurably too wide and is not claimed here.
"""

import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from effective.lint import (
    check_deps_source,
    check_layer_source,
    check_source,
    declares_layer,
    is_layer_role,
    is_workflow_role,
    seam_forbidden_for,
    yields_effect_op,
)

_RUFF_CONFIG_MARKER = "[tool.ruff]"


@cache
def find_ruff_config() -> Path | None:
    """The repo's own ruff config — line length, select set, target version — or None.

    Searched upward rather than assumed, so this works from a package member, a worktree, or a
    checkout at any depth.

    **Resolved on CALL and cached, not at import.** A module-level search that raised would make
    `import effective.coding.gate` fail on any box that is not a source checkout (a wheel in
    site-packages has no repo `pyproject.toml` above it), deciding a packaging property by an
    import side effect.

    Returns None rather than raising **because the two callers want different answers and the
    difference is real**: this module JUDGES, so it refuses (below), while
    `edits.structural.apply_structural` merely TIDIES a rewrite with `--fix` and ruff's defaults
    are an acceptable tidy. Putting the search here and the policy at each call site is what makes
    that a choice rather than two hand-written loops that happen to differ."""
    for parent in Path(__file__).resolve().parents:
        cfg = parent / "pyproject.toml"
        if cfg.exists() and _RUFF_CONFIG_MARKER in cfg.read_text():
            return cfg
    return None


_RUFF_LINE = re.compile(r":\d+:\d+:\s+(?P<code>[A-Z]+\d+)\s+(?P<msg>.*)")
_ALLOWED_WHITESPACE = frozenset("\n\t\r")


def ruff_diagnostic(content: str, name: str) -> str | None:
    """Run ruff over `content` with the repo's config and **no `--fix`**.

    Written to a real temp file rather than piped through `--stdin-filename`, because stdin MISSES
    module-level rules like `F822` that `ruff check <file>` catches — so this way the verdict
    equals the command a human would run. The file lives outside the repo and keeps the edited
    file's basename, so `__init__.py`-specific handling still matches.

    No `--fix` on purpose: the model must produce a *mergeable* edit, not one auto-fixed into
    shape. Auto-fixing here would let the gate launder work the model did not do."""
    ruff = shutil.which("ruff")
    if ruff is None:
        # REFUSE, never "clean". A verdict that reads PATH is not a verdict: the same content
        # would be refused on a box with ruff and committed on one without, and committed is the
        # answer that survives. `apply_structural` answers this the same way one rung up
        # ("ast-grep unavailable"), and delivering a `ToolError` into a workflow rests on this
        # function being re-derivable rather than environment-shaped.
        return "lint [ruff] ruff is not on PATH, so this content was never vetted"
    config = find_ruff_config()
    if config is None:
        # Same refusal, same reason, one rung along: ruff's DEFAULTS are not this repo's rules
        # (line length 88 vs 99, a different select set), so a verdict reached without the config
        # would pass content `just lint` rejects. Absence must not read as clean.
        return "lint [ruff] no repo ruff config found, so this content was never vetted"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / Path(name).name
        f.write_text(content)
        proc = subprocess.run(
            [
                ruff, "check", "--quiet", "--no-fix", "--output-format=concise",
                "--config", str(config), str(f),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )  # fmt: skip
    if proc.returncode == 0:
        return None
    line = next((ln for ln in proc.stdout.splitlines() if _RUFF_LINE.search(ln)), "")
    if (m := _RUFF_LINE.search(line)) is not None:
        return f"lint [ruff {m.group('code')}] {m.group('msg').strip()}"
    return f"lint [ruff] {(proc.stdout.strip() or proc.stderr.strip())[:160]}"


def nonprintable_diagnostic(content: str) -> str | None:
    """The first control byte that is not legitimate whitespace — the floor every other gate
    misses. `str.isprintable()` accepts all normal Unicode, so an em-dash is fine and a DEL is
    not."""
    line = 1
    for ch in content:
        if ch == "\n":
            line += 1
        elif ch not in _ALLOWED_WHITESPACE and not ch.isprintable():
            return f"lint [fidelity] non-printable character U+{ord(ch):04X} at line {line}"
    return None


def static_diagnostic(content: str, name: str) -> str | None:
    """The first reason `content` would not merge, or `None`.

    Ordered so the diagnostic is the useful one: fidelity (any file), then parse, then
    `effective.lint` role-dispatched exactly as `just lint` dispatches it, then ruff.

    **Role dispatch is by NAME *or* by CONTENT, and the second disjunct is why this gate can see a
    file the machine WRITES.** `just lint` dispatches on membership in a curated role tuple, which
    is right for a repo whose files someone added on purpose and useless here: a name the model
    invented is in no tuple, so identical bytes were refused under `src/effective/react.py` and
    accepted under `src/effective/whatever_it_called_it.py` — the closing predicate blind to the
    boundary it exists to protect. `yields_effect_op` / `declares_layer` are the same operational
    definitions `--role-coverage` and the layer lint already own, asked of the content instead of
    the path, so the two cannot drift into disagreeing about what a workflow is. The name arm stays
    because a curated entry is a DECISION (a file that is workflow-role but whose first op yield
    has not been written yet is still scanned)."""
    if (fidelity := nonprintable_diagnostic(content)) is not None:
        return fidelity
    if not name.endswith(".py"):
        return None  # the Python half applies to Python source only
    try:
        compile(content, name, "exec")
    except SyntaxError as e:
        return f"lint [syntax] {e.msg} (line {e.lineno})"
    violations: list[Any] = []
    if is_workflow_role(name) or yields_effect_op(content):
        violations += check_source(content, name)
    if is_layer_role(name) or declares_layer(content):
        violations += check_layer_source(content, name)
    violations += check_deps_source(content, name, seam_forbidden_for(Path(name)))
    if violations:
        v = violations[0]
        return f"lint [{v.rule}] {v.message}"
    return ruff_diagnostic(content, name)


def static_merge_gate(tree: Mapping[str, str]) -> str | None:
    """The first file in a whole tree that would not merge, or `None`.

    This is the closing predicate's STATIC half: "the run closed" means parse-clean, lint-clean and
    ruff-clean — deliberately stronger than the import-clean predicate a nano run once closed
    against, and deliberately weaker than `just check`, which also runs `ty` and the suite."""
    for name in sorted(tree):
        if name.endswith(".py") and (d := static_diagnostic(tree[name], name)) is not None:
            return f"{name}: {d}"
    return None
