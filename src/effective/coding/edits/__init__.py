"""The edit-scheme ladder: text and structural, and why the rung decides where work runs.

The rungs are monotone in how much of the world an edit needs, and that fixes the sandbox
boundary: the rung an edit needs decides whether it can happen inside a sandbox at all.

| rung       | knows               | tool          | renaming `x` beside a string `"x"`   |
|------------|---------------------|---------------|--------------------------------------|
| text       | the file's bytes    | `re`, `str`   | rewrites the string too: wrong       |
| structural | the file's parse    | `ast_grep_py` | rewrites only the `x` nodes          |

> **The sandbox boundary and the nominal/structural boundary are the same boundary.** Text needs
> bytes and runs anywhere; structural needs a parse, and `import ast` is a `ModuleNotFoundError`
> inside Monty.

**So the structural host call is the default, and a text program must argue for itself.** A
sandbox makes the nominal scheme cheap (`re` works) and the structural one impossible, so the path
of least resistance is regex over code, this repo's named defect class.

**What the model emits is an intent, never the result.** Its whole surface is an ast-grep pattern;
locating, rewriting, formatting and validating are deterministic Python derived from it, so a
replay re-binds the recorded intent and never re-runs the tool. That keeps an editor inside the
determinism boundary.
"""
