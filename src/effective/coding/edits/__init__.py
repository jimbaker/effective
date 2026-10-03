"""The edit-scheme ladder: text, syntactic, semantic — and why the rung decides where work runs.

Three rungs, and they are monotone in how much WORLD an edit needs. That is not a taxonomy for its
own sake; it fixes the sandbox boundary, because the rung you need decides whether the edit can
happen inside a sandbox at all.

| rung | knows | tool | the discriminating rename |
|---|---|---|---|
| text | the file's **bytes** | `re`, `str` | renames both scopes — wrong |
| syntactic | the file's **parse** | `ast_grep_py` | 4 nodes, both scopes — wrong |
| **semantic** | **binding + lexical scope** | **`jedi`** | 2 references, `f` only — right |

Measured on `rename x in f() where g() has its own x and a string "x"`. Only the semantic rung
gets it right, because only it knows what a name is BOUND to rather than how it is spelled.

> **The sandbox boundary and the nominal/structural boundary are the same boundary.** Text needs
> bytes and runs anywhere; syntactic needs a parse (`import ast` is a `ModuleNotFoundError` inside
> Monty); semantic needs the whole project — bindings, imports, inference — and cannot run remotely
> at all.

**So the structural/semantic host call is the DEFAULT, and a text program is the exception that
must argue for itself.** The hazard is that a sandbox makes the *nominal* scheme cheap (`re` works)
and the structural one impossible, so the path of least resistance is regex-over-code — this
repo's named defect class. Defaulting the other way is the whole point of the ladder.

**What the model emits is an INTENT, never the result.** For the structural rung the model's entire
surface is an ast-grep pattern; locate, rewrite, format and validate are deterministic Python
re-derived from it. For the semantic rung it is a position and a new name; jedi computes the diff.
Both are pure functions of their inputs, so a replay re-binds the recorded intent and never
re-runs the tool — which is what keeps an editor inside the determinism boundary.
"""

from effective.coding.edits.semantic import (
    SemanticEdit,
    SemanticError,
    declared_arms,
    references,
    rename,
)
from effective.coding.edits.structural import (
    StructuralEdit,
    StructuralRule,
    apply_structural,
    structural_matches,
)

__all__ = [
    "SemanticEdit",
    "SemanticError",
    "StructuralEdit",
    "StructuralRule",
    "apply_structural",
    "declared_arms",
    "references",
    "rename",
    "structural_matches",
]
