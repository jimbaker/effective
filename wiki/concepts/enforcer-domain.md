# A gate is bounded by what it scans

Before trusting a green, ask what the enforcer's grammar actually reaches. The sentence describing
a gate will quietly claim more than the gate covers, and **the enforcer's grammar is its domain**.

A denylist is bounded by what it *enumerates*; a gate is bounded by what it *scans*. Both fail the
same way, and the fix is the same move: key on the **resolved** thing, not on the text used to
reach it.

## Worked cases, each found the expensive way

| the enforcer | its prose said | its grammar covered | how it surfaced |
|---|---|---|---|
| `op_key`'s reserved prefixes | author names stay out of substrate namespaces | the op-arm tags, **not** the authority namespaces | widened once, then **deleted entirely**: a `Step`'s key opens with its own `step` arm, so `step;ledger:x` is disjoint from `ledger:x` by construction and there is nothing left to enumerate |
| `just docs-check` | dangling documentation links fail | repo-rooted paths, **not** the bare-basename form the wiki itself uses | five commits of file moves passed green while documents were left citing paths in the bare-basename form |
| `--role-coverage`'s `_EFFECT_MODULES` | every effect module is covered | a list of **import spellings of one thing** | widened for `from effective import ask_llm`, then missed `from effective.react import run_agent` two commits later |
| `--forged-join` | a key rebuilt by joining rendered terms | one spelling, **not the literal `";".join(...)` its own docstring uses to define the defect** | a reviewer wrote five forging functions; the gate saw one |
| `scripts/fstring_sweep.py` | no new f-string leaks into `src/` | `(path, statics)`, deduped, so a new f-string in an already-baselined file is free | measured: two new f-strings, one a key mint and one a bare flatten inside the t-string processor, both `0 new, 0 stale` |
| `just docs-check`, again | a citation that names nothing fails | links; **a backticked `-task` slug is not a link** | `just check` passed with a slug cited from `src/`, the layer that is current by contract |
| `just docs-check`, a fourth time | a citation resolves | the **path**, never the LINE. `file.py:1513` grades identically to `file.py:1` | **not yet bitten, and the count is the point**: `docs/effective-design.md` carries 12 such citations into `absurd.py`, `layers.py` and `base.py`, all three edited by an open branch. Source moves, the number stays, the gate stays green. Re-verifying line citations is a manual pass |
| `just docs-check`, a third time | a citation resolves | whatever is on **this machine**: `resolves()` asked the filesystem, so a gitignored artifact resolved here and in no clone | CI's `check` job died at `docs-check` before reaching a test, on every run for two months, while the same commit read `0 new, 0 stale` locally |

## The sub-shape that keeps recurring

**A domain enumerated as spellings of a single referent will always be one spelling short**,
because the referent grows new spellings and nothing links them.

And when a domain keeps coming up one entry short, ask whether the property it tests is **closed
under its own application**. `--role-coverage`'s was: `agent/bench_sweep.py` writes
`yield from improve(...)`, and `effective.improve` is an effect module only because it yields
`gather`/`scoped` itself. So the domain is a **fixpoint**, not a bigger list:
`lint._effect_module_closure` seeds with `_EFFECT_MODULES` and closes over the scanned files,
adding eight modules.

## The domain can be the MACHINE, not the grammar

Every case above is a grammar too narrow for its subject. The last row is not: that grammar was
fine and its **resolver** consulted the wrong world. `(REPO / path).exists()` answers about the
working tree, which holds build output, hook output and everything gitignored, so the verdict was a
function of who ran it.

So the question has two halves. *What does the grammar match* finds the rows above; **what would a
different machine answer** finds this one, and the answer to it is a command, not a reading: run
the gate in a fresh clone and diff the verdict. It is the same move the table keeps making, keyed
on the **resolved** thing, one level out: what is resolved here is not a name but a world.

Naming the world creates a second surface, and it needs its own check. A declared exemption for a
generated artifact is a hole in the domain, so the declaration is checked too: the file must still
be untracked, something must still cite it, and its **producer** must still exist. The first two
cannot fire for the entries that exist; only the third can, which is worth knowing about any
exemption you write.

## And the domain can be the RENDERING

A third arm, and it is the one a REPORT fails at rather than a gate. The grammar can be right, the
resolver can consult the right world, and the instrument can still under-report because of how it
DISPLAYS what it found.

`just key-view` sorts every registered key shape by template, and exists to make duplication
visible. A `Shape` records one `site` and arbitrarily many `sites`; it printed the first. So **103
producing sites sat behind 82 rows**, and the 21 it hid were exactly the finding: 16 templates are
minted at more than one place, `tool:{}` among them, at `api.py:53` and an inlined twin at
`coroutine_api.py:51`. Nothing was mis-scanned. The view was lossy at the last step.

So the question has three halves now. *What does the grammar match* finds the table above; *what
would a different machine answer* finds the resolver row; **what did the renderer collapse** finds
this one. A view that shows one representative per group cannot show you a group's size, and if
group size is the finding, the instrument is answering a different question than the one it was
built for.

## Two more things a zero can mean

**A zero is a claim about the enumerator before it is a claim about the code.** Every one of the
four `whouses.py` defects reported absence: module-level `def`s swept as `$X.name` reported
`0 syntactic candidates`: 819 of them, against 332 constants refused outright, so the tool
addressed 371 of the 1,522 module-level definitions under `src/effective` and `src/agent`, and a
PEP 634 keyword pattern was invisible, 159 sites in `src/effective`. A "zero uses, delete it" sweep
on that enumerator would have systematically proposed deleting the code this repo most prefers to
write. Fixed by enumerating with ast-grep RULES instead of patterns; `tests/test_whouses.py` pins
each.

**And the enumerator's domain was ast-grep's pattern DSL, not ast-grep.** The filed diagnosis said
`inside: {kind: class_pattern}` "returns zero", so the parent had to be walked by hand. It returns
159 of 159 with `stopBy: end`; the default is `neighbor`. A tool's stated ceiling is a claim too,
and this one blinded the sweep to the construct the repo writes most for the whole twelve days the
tool had existed.

**UNRESOLVED is a third outcome**, never folded into "not a use". An inference miss and a genuine
non-use are different facts, and merging them makes a worse `grep` wearing a type checker's
clothes.

## The corollary for a claim

A measurement licenses a claim only over the **domain it covered**. So: an engine more precise than
the reference can launder an unproved property into a green test. Ask what a passing test *relies
on*, not just that it passes. And **an instrument is a claim; the one you built this hour is the
one you are least able to check.** You find a green-and-wrong instrument by MOVING what it
describes, not by running it.
