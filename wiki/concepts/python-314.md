# Python 3.14 is a showcase here

**When 3.14 syntax looks wrong, suspect a feature before "fixing" it.** This repo is
written by PEP 750's lead author and leans into the new language surface on purpose. If it compiles, imports, and
`uvx ty check` passes, it is almost certainly intentional: verify against the 3.14 changelog
rather than correcting it to an older idiom.

| PEP | what it gives | where it shows |
|---|---|---|
| **750: t-strings** | the data axis: a `Template` whose interpolations are typed I/O channels | `effective.channels`, [`examples/first_workflow.py`](../../examples/first_workflow.py), `compose_key`, the psycopg boundary. See [concepts/flatten](flatten.md) |
| **758: `except`/`except*` without parentheses** | `except ValueError, TypeError:` catches **both**: valid 3.14, *not* legacy Python-2 `except E, name:` | `agent/bracket.py` |
| **649/749: deferred annotation evaluation** | annotations are not evaluated at definition time, so forward refs and heavier typing read naturally | everywhere; no `from __future__ import annotations` |

The interpreter is pinned exactly at `.python-version`, because it was the one toolchain here that
floated. A green gate is green **on the interpreter it ran on**.

Do not reach for 3.15 syntax before 3.15 is the pinned interpreter.
