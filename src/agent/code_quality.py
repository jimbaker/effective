"""Code quality as ASI — the CLAUDE.md rubric, mechanized and pluggable.

Quality for code in this repo is not vibes; it is the project's own gates. This runs
**static** checkers over candidate code and returns the tool output as the **ASI** (GEPA's
gradient): the ruff/ty message is exactly what the proposer reflects on, so a debug loop
with `quality_floor=1.0` fixes the bug and THEN cleans up whatever the tools flag.

Two properties make this safe and general:

- **Static, so no sandbox.** Every checker *reads* the code (ruff, ty); it never runs it.
  So `check_code` is safe on untrusted, model-improvised code with no container. *Executing*
  code beyond the Monty subset is the dangerous part, and needs an execution tier.
- **Pluggable + language-extensible.** A `Checker` is just `files -> (violations, notes)`.
  `ruff_checker`/`ty_checker` cover Python; another language's linter/typechecker plugs in
  the same way. `make_code_quality((ruff_checker(), ty_checker()))` composes them.

The connection to `run_code`: the code a `run_code` call runs (an improvised string or a
pinned skill script's source) is a plain code string, and the Monty subset is valid Python,
so ruff/ty apply. `check_code(code)` is that seam: a quality/ASI reading
of run_code's code, the signal a run_code improve loop would tighten against.

Handler-side (behind the `check_quality` op); the `QualityReport` is recorded, so replay
reuses it — the tools run once.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from agent.debug import QualityFn, QualityReport

# a Checker writes nothing — it reads already-written files and reports (violations, notes)
type Checker = Callable[[Sequence[Path]], tuple[int, str]]

_RUFF_LINE = re.compile(r":\d+:\d+:")  # one ruff concise line per violation
_TY_ERROR = re.compile(r"error\[")  # one ty concise line per diagnostic


def _venv_bin(name: str) -> str | None:
    """A tool in the active venv, next to the interpreter (where `uv run` puts it) — not on
    the bare PATH."""
    candidate = Path(sys.executable).with_name(name)
    return str(candidate) if candidate.exists() else shutil.which(name)


def ruff_checker(
    *, select: Sequence[str] = ("F", "E9"), config: str | Path | None = None
) -> Checker:
    """Lint/idioms via ruff. Default rules (F = pyflakes: unused/undefined; E9 = syntax) are
    a stable subset; pass ``config=<repo>/pyproject.toml`` for the project's full lint gate."""
    ruff = _venv_bin("ruff") or "ruff"

    def check(paths: Sequence[Path]) -> tuple[int, str]:
        cmd = [ruff, "check", "--output-format=concise", "--select", ",".join(select)]
        if config is not None:
            cmd += ["--config", str(config)]
        r = subprocess.run([*cmd, *map(str, paths)], capture_output=True, text=True)
        n = len(_RUFF_LINE.findall(r.stdout))
        return n, ("" if n == 0 else r.stdout.strip())

    return check


def ty_checker() -> Checker:
    """Types via ty (`uvx ty check`). Best for **self-contained** code — an isolated file
    importing project modules can't resolve them, but run_code Monty scripts are typically
    self-contained. Opt-in (not in the default rubric)."""

    def check(paths: Sequence[Path]) -> tuple[int, str]:
        r = subprocess.run(
            ["uvx", "ty", "check", "--output-format=concise", *map(str, paths)],
            capture_output=True,
            text=True,
        )
        n = len(_TY_ERROR.findall(r.stdout))
        return n, ("" if n == 0 else r.stdout.strip())

    return check


def make_code_quality(checkers: Sequence[Checker] | None = None) -> QualityFn:
    """A `quality_fn` that runs each checker over the candidate files and scores it: 1.0 when
    clean, degrading with total violation count; the notes are the tools' own output (the
    ASI). Default rubric is ruff alone (reliable standalone); compose ``ty_checker()`` (or
    another language's checker) for more."""
    active = tuple(checkers) if checkers is not None else (ruff_checker(),)

    def quality(files: Mapping[str, str]) -> QualityReport:
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, content in files.items():
                p = Path(d) / Path(name).name
                p.write_text(content)
                paths.append(p)
            total, notes = 0, []
            for checker in active:
                n, note = checker(paths)
                total += n
                if note:
                    notes.append(note)
        return QualityReport(
            score=1.0 if total == 0 else 1.0 / (1.0 + total), notes="\n".join(notes)
        )

    return quality


def check_code(
    code: str, *, name: str = "script.py", checkers: Sequence[Checker] | None = None
) -> QualityReport:
    """The `run_code` connection: a quality/ASI reading of the code a `run_code` call runs —
    an improvised string or a pinned skill script's source (both plain code strings; the
    Monty subset is still valid Python, so ruff/ty apply). The returned notes are the ASI a
    run_code improve loop reflects on to tighten the next attempt. This is **static** (no
    execution), so it is safe on untrusted code with no sandbox. For a non-Python run_code
    target, pass that language's checkers.

    (Refinement, designed-for: run_code injects names — ``inputs``/``functions``/``actions`` —
    that a standalone ruff pass flags as F821 undefined. A `provides` param that declares
    them for the checker is the clean follow-up; until then, check self-contained code or
    select rules that skip F821.)"""
    return make_code_quality(checkers)({name: code})
