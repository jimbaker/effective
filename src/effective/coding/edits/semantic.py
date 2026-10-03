"""Rung 3, the semantic editor: jedi, which knows what a name is BOUND to.

This is the ladder's top rung as something the machine can call.

**What the rung buys, measured on the discriminating case**: rename `x` in `f()`, where `g()` has
its own `x` and there is a string `"x"`:

    text        `re`/`str`     renames both scopes and the string       wrong
    syntactic   `ast_grep_py`  4 nodes across both scopes               wrong
    semantic    `jedi`         2 references, `f` only                   right

And on the reading side, on a real query in this repo: `grep -rn op_key src/` returns 104 hits,
roughly half of them prose; `jedi.get_references` returns 23 actual references with exact
`file:line`. A resolver removes the transcription error at its source.

**Its limits, stated rather than discovered.** A resolver's line number is exact at query time and
rots the moment the file is edited above it, as grep's does. And a reference living in DATA — a
name inside a string, a composed key, a `getattr` — is visible to grep and invisible to jedi. That
form is not rare here: `ast-grep run --lang python --pattern 'compose_key($$$)' src/
--json=compact | jq length` reports the identities built that way, none of which jedi can reach.
So "jedi finds everything" is false, and the two instruments have different blind spots rather
than a better and a worse one.

**`ty` is the type authority, not jedi.** jedi does its own inference, and where the two disagree
the repo's checker wins. `declared_arms` is therefore offered as an *analysis* a worker may use to
propose an edit, never as a verdict: the verdict comes from running `ty`. A semantic editor that
trusted jedi's types over `ty`'s would be a second type system nobody ratified.

**The edit is a DIFF, and that is deliberate.** `rename` returns unified-diff text rather than
writing files, so the intent is recordable, reviewable and replayable, and applying it stays a
separate, explicit act.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import jedi
from jedi.api.classes import Name


class SemanticError(RuntimeError):
    """jedi could not answer — an unresolvable position, a syntax error, a name it cannot bind.

    Raised rather than returned-as-empty, because "no references" and "I could not tell" are
    different answers and collapsing them is how a rename silently does nothing."""


@dataclass(frozen=True, slots=True)
class Reference:
    """One resolved reference: where it is, and what it is called there."""

    path: str
    line: int
    column: int
    name: str


@dataclass(frozen=True, slots=True)
class SemanticEdit:
    """A rename, as an intent plus the diff it derives. `diff` is unified-diff text over the
    project, so a reviewer reads exactly what would land and a replay re-binds it unchanged."""

    path: str
    line: int
    column: int
    new_name: str
    diff: str
    changed: tuple[str, ...]


def _inside(root: Path, path: str | Path) -> bool:
    """Does `path` resolve inside `root`? Resolving is the check, since a symlink or `..` leaves
    the project while its spelling stays in it, and the file's content would reach a recorded
    op result."""
    try:
        return (root / path).resolve().is_relative_to(root)
    except (OSError, ValueError) as unresolvable:
        raise SemanticError(f"{path!r} does not resolve: {unresolvable}") from unresolvable


_INTERPRETER = tuple({Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()})
"""Where the interpreter's own modules live: its standard library and installed packages."""


def _answerable(root: Path, definition: Name) -> bool:
    """Is `definition` the project's, the language's or the interpreter's? Anything else names
    another file's content."""
    if definition.in_builtin_module():
        return True
    if definition.module_path is None:
        return False
    return _inside(root, definition.module_path) or any(
        _inside(prefix, definition.module_path) for prefix in _INTERPRETER
    )


def _script(project_root: str | Path, path: str | Path) -> jedi.Script:
    root = Path(project_root).resolve()
    if not _inside(root, path):
        raise SemanticError(f"{path} is outside the project")
    full = (root / path).resolve()
    if not full.exists():
        raise SemanticError(f"no such file: {full}")
    return jedi.Script(path=str(full), project=jedi.Project(str(root)))


def references(
    project_root: str | Path, path: str | Path, line: int, column: int
) -> tuple[Reference, ...]:
    """Every reference to the name at this position inside the project: bindings, not spellings.
    jedi also answers with the definitions it resolved to elsewhere, and those are left out."""
    root = Path(project_root).resolve()
    try:
        found = _script(project_root, path).get_references(
            line=line, column=column, include_builtins=False, scope="project"
        )
    except Exception as e:  # jedi raises a wide family; the caller wants one type
        raise SemanticError(f"could not resolve references at {path}:{line}:{column}: {e}") from e
    return tuple(
        Reference(path=str(r.module_path), line=r.line, column=r.column, name=r.name)
        for r in found
        if r.module_path is not None
        and r.line is not None
        and r.column is not None
        and _inside(root, r.module_path)
    )


def rename(
    project_root: str | Path, path: str | Path, line: int, column: int, new_name: str
) -> SemanticEdit:
    """Rename the BINDING at this position, project-wide, and return the diff.

    Scope-correct by construction: a same-spelled name in another scope, and the same characters
    inside a string, are untouched — which is the entire difference between this rung and the two
    below it."""
    if not new_name.isidentifier():
        raise SemanticError(f"{new_name!r} is not a Python identifier")
    root = Path(project_root).resolve()
    try:
        refactoring = _script(project_root, path).rename(
            line=line, column=column, new_name=new_name
        )
        diff = refactoring.get_diff()
        changed = tuple(sorted(str(p) for p in refactoring.get_changed_files()))
    except SemanticError:
        raise
    except Exception as e:
        raise SemanticError(f"could not rename at {path}:{line}:{column}: {e}") from e
    if outside := [changed_path for changed_path in changed if not _inside(root, changed_path)]:
        raise SemanticError(
            f"renaming at {path}:{line}:{column} would change {len(outside)} file(s) outside "
            f"the project"
        )
    return SemanticEdit(
        path=str(path), line=line, column=column, new_name=new_name, diff=diff, changed=changed
    )


def declared_arms(project_root: str | Path, path: str | Path, alias: str) -> tuple[str, ...]:
    """The arms of a declared union alias — FINALIZE's analysis, and what no lower rung can do.

    Converting an `isinstance` chain into a decision table closed by `assert_never` needs to know
    what a union DECLARES, which is a fact about types and not about shape. Measured: jedi resolves
    a PEP 695 `type U = A | B` to `('A', 'B')`.

    **Advisory, not authoritative** — see the module docstring. Use it to propose the arms; let
    `ty` say whether the result is total. A union with an arm defined outside the project and the
    interpreter is refused whole, since fewer arms would be a wrong answer."""
    root = Path(project_root).resolve()
    script = _script(project_root, path)
    for name in script.get_names(all_scopes=True, definitions=True):
        if name.name == alias:
            try:
                arms = name.infer()
            except Exception as e:
                raise SemanticError(f"could not infer arms of {alias!r}: {e}") from e
            if outside := [arm for arm in arms if not _answerable(root, arm)]:
                raise SemanticError(
                    f"{alias!r} has {len(outside)} arm(s) defined outside the project"
                )
            return tuple(arm.name for arm in arms if arm.name)
    raise SemanticError(f"{alias!r} is not defined in {path}")
