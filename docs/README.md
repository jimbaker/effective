# `docs/` and `wiki/`

**A document in `docs/` or `wiki/` is current. If it disagrees with source, that is a bug.**

The two directories split by kind, and both are held to that contract.

| directory | holds | shape |
|---|---|---|
| `docs/` | deliverables a reader works through: the first workflow, the concepts tour, the design note, the repertoire axis | long documents with numbered sections |
| `docs/adr/` | the architecture decision records: one decision each, its context, the alternatives, and what is built | cited by number, indexed in `wiki/index.md` |
| `wiki/` | the arguments behind the design, one concept a page, interlinked by `[[concepts/...]]` | short pages, rewritten in place when the truth changes |

`wiki/index.md` catalogs both. A filename carries no date, because a document here describes the
tree as it is; what changed and why goes in the commit message.

## What a reader opens first

```
run something first? ─────────────► first-workflow.md        (one command: run, resume, guardrail)
                                       │
new to Effective? ────────────────► intro.md                 (the model in ten minutes)
                                       │
the concepts in depth? ───────────► effective-101.md         (the concepts, in order)
                                       │
need the semantics? ──────────────► effective-design.md      (ops, small-step rules, keys)
                                       │
looking for WHERE something is? ──► wiki/concepts/architecture.md
                                       │
looking for WHY it is that way? ──► wiki/index.md            (the concept pages, the ADR index)
```

## The gates

Two gates guard citations, and neither sees what the other does. `just docs-check` grades
repo-rooted paths in `README.md`, `CLAUDE.md`, `docs/` and `wiki/`: a path that names no file
fails. `just wiki-lint` grades `[[links]]` and orphan pages. Both run inside `just check`.
