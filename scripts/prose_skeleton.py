"""Prove an edit touched prose and nothing else — the two invariants a de-essaying pass needs.

`doc_bloat` and `verdict_words` say WHERE to look and WHAT reads badly. Neither can tell you
whether the edit you then made moved any code. These two can:

- **skeleton** — parse the file, blank every string-literal expression, `ast.dump` the result.
  Equal before and after means no statement moved, no name changed, no argument was added. It is
  blind by construction to exactly the thing a prose edit is allowed to change.
- **fingerprints** — hash each prose unit by its location. The set of units whose hash changed is
  the set the edit touched, so a pass that claims to have rewritten three docstrings can be held
  to three.

    uv run python scripts/prose_skeleton.py --save before.json src/effective/keys/processor.py
    # ... edit prose ...
    uv run python scripts/prose_skeleton.py --check before.json src/effective/keys/processor.py

**Attribute docstrings count.** The bare string after an assignment is a prose unit this repo uses
heavily, and the corpus census that missed them undercounted by 182 units and 14,699 words.
`ast.get_docstring` does not see them, so this walks statement bodies instead.

**Comments are NOT covered, and the hole is load-bearing rather than an oversight.** A `#` comment
is not an AST node, so neither invariant can see one change — and `parked.py`'s single largest
history block was a forty-line comment. `--comments` reports a `tokenize` count beside the AST
work so a pass at least knows whether comment volume moved; locating a comment edit needs
ast-grep's `kind="comment"`, which is a different rung.
"""

import argparse
import ast
import hashlib
import json
import pathlib
import sys
import tokenize
from typing import TypedDict


def _blank(tree: ast.AST) -> ast.AST:
    """Replace every string-literal EXPRESSION with a fixed marker.

    Expression, not constant: a string used as a VALUE (`kind="comment"`, a dict key, a default
    argument) is code, and blanking it would make the skeleton blind to a real edit. The
    discriminator is the same one `doc_bloat` uses to measure code lines — an `ast.Expr` whose
    value is a `Constant` is a statement whose whole purpose is to sit there being a string."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if _prose_text(stmt) is not None:
                    body[i] = ast.Expr(value=ast.Constant(value="<prose>"))
    return tree


def _prose_text(stmt: object) -> str | None:
    """The text of a prose statement, or `None` if this statement is not one.

    A `match` rather than an `isinstance` chain, and the pattern says the whole rule in one line:
    an `Expr` wrapping a string `Constant` is a statement whose entire purpose is to be prose. A
    string used as a VALUE is an `Assign`, a `keyword` or an argument default, so it does not
    match and stays visible to the skeleton — which is what makes changing `"answer"` to
    `"answer_TYPO"` a code change here rather than a prose one."""
    match stmt:
        case ast.Expr(value=ast.Constant(value=str() as text)):
            return text
        case _:
            return None


def skeleton(path: pathlib.Path) -> str:
    """The file with all prose blanked, dumped. Two files with the same skeleton differ only in
    prose."""
    return ast.dump(_blank(ast.parse(path.read_text())))


def units(path: pathlib.Path) -> dict[str, str]:
    """Every prose unit, keyed by a LOCATION that survives an edit to its own text.

    The key is the enclosing definition's qualified name plus an index — `Key.parse#0` — rather
    than a line number, because a prose edit changes every line number below it and a location that
    moves cannot be compared across the edit. Module-level units key off `<module>`."""
    tree = ast.parse(path.read_text())
    found: dict[str, str] = {}
    _walk(tree, "<module>", found)
    return found


def _walk(node: ast.AST, scope: str, found: dict[str, str]) -> None:
    body = getattr(node, "body", None)
    if isinstance(body, list):
        index = 0
        for stmt in body:
            if (text := _prose_text(stmt)) is not None:
                found[f"{scope}#{index}"] = hashlib.sha256(text.encode()).hexdigest()[:16]
                index += 1
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            _walk(child, f"{scope}.{child.name}" if scope != "<module>" else child.name, found)


def comment_words(path: pathlib.Path) -> tuple[int, int]:
    """`(comments, words)` via `tokenize` — the corpus the AST invariants cannot reach."""
    with path.open("rb") as handle:
        comments = [
            tok.string
            for tok in tokenize.tokenize(handle.readline)
            if tok.type == tokenize.COMMENT
        ]
    return len(comments), sum(len(c.lstrip("#").split()) for c in comments)


def _paths(roots: list[str]) -> list[pathlib.Path]:
    """A FILE argument is expanded here rather than left to `rglob`, which matches nothing on a
    file path and returns silently — the failure that put a false measured claim in four documents
    (`doc_bloat.py`'s own header records it). Both report tools now take a file; so does this."""
    out: list[pathlib.Path] = []
    for root in roots:
        base = pathlib.Path(root)
        out.extend([base] if base.suffix == ".py" else sorted(base.rglob("*.py")))
    return out


class Snapshot(TypedDict):
    """One file's before-state. A `TypedDict` rather than `dict[str, object]` because the two
    fields have different types and `len(entry["units"])` is a `ty` error against the loose one —
    which is the loose annotation costing a real diagnostic rather than merely reading vaguely."""

    skeleton: str
    units: dict[str, str]


def snapshot(roots: list[str]) -> dict[str, Snapshot]:
    return {
        str(path): Snapshot(
            skeleton=hashlib.sha256(skeleton(path).encode()).hexdigest(),
            units=units(path),
        )
        for path in _paths(roots)
    }


def _check(before: dict[str, Snapshot], current: dict[str, Snapshot]) -> int:
    """Compare a snapshot against the working tree. Non-zero if any edit was not prose-only."""
    failed = False
    for path, entry in current.items():
        prior = before.get(path)
        if prior is None:
            print(f"NEW FILE (not in snapshot): {path}")
            failed = True
            continue
        if prior["skeleton"] != entry["skeleton"]:
            print(f"SKELETON CHANGED: {path} — this edit moved CODE, not just prose")
            failed = True
        changed = sorted(
            name
            for name in set(prior["units"]) | set(entry["units"])
            if prior["units"].get(name) != entry["units"].get(name)
        )
        print(f"  {path}: {len(changed)} prose unit(s) changed")
        for name in changed:
            print(f"      {name}")
    for path in set(before) - set(current):
        print(f"MISSING (in snapshot, not read now): {path}")
        failed = True
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", metavar="FILE", help="write a snapshot and exit")
    ap.add_argument("--check", metavar="FILE", help="compare the working tree against a snapshot")
    ap.add_argument("--comments", action="store_true", help="also report tokenize comment counts")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    current = snapshot(args.paths)

    if args.comments:
        for path in _paths(args.paths):
            count, words = comment_words(path)
            print(f"  {path}: {count} comments, {words} words")

    if args.save:
        pathlib.Path(args.save).write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"saved {len(current)} file(s) to {args.save}")
        return 0

    if not args.check:
        for path, entry in current.items():
            print(f"  {path}: {len(entry['units'])} prose units")
        return 0

    return _check(json.loads(pathlib.Path(args.check).read_text()), current)


if __name__ == "__main__":
    sys.exit(main())
