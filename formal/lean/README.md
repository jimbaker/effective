# Effective — Lean 4 formalization

The **pure-lemma half** of the Effective formal-verification investigation: the
facts true of the *algebra of names and rules*, independent of any interleaving
(the reachability/temporal half is Quint, in `formal/quint/`). The goals are T1–T3,
the proof DAG is the `L1.x`/`L2.x`/`L3.x` nodes below, and Lean owns the pure facts
while Quint owns reachability and interleavings.

## Status

| Node | What | Tactic | Axioms | Where |
|---|---|---|---|---|
| L1.1 | `Key` type + `key` / `keyBad` encoders | — | — | `Effective/Keys.lean` |
| L1.2 | regression: the cross-gather collision; `keyBad` not injective | `decide` | none | `Effective/Keys.lean` |
| L1.3 | `Function.Injective key` — the correct encoder never aliases | induction | `propext` | `Effective/Keys.lean` |
| L1.4 | `branch_disjoint` — different `(g,i)` frames never collide | `key_injective`+`simp` | `propext` | `Effective/Keys.lean` |
| L1.5 | `gather_naming_injective` — fan-out naming injective (⨄ defined; the Quint export) | `key_injective`+`simp` | `propext` | `Effective/Keys.lean` |
| L1.6 | scope grammar defs: `flatten` + the delimiter-free `flattenBad` foil | — | — | `Effective/Scopes.lean` |
| L1.7 | regression: `flattenBad` aliases `("a","12")`/`("a1","2")`; `flatten` separates them | `decide` | none | `Effective/Scopes.lean` |
| L1.8 | `flatten_injective_on` — the flattening is injective on the `WF` domain | induction | `propext` | `Effective/Scopes.lean` |
| T2 | `step_deterministic` — the transition relation is a partial function (I6) | `cases`+`simp_all` | `propext` | `Effective/Step.lean` |
| L2.5 | `replay_stable` — recorded checkpoint forces replay; oracle never consulted | `cases`+`simp_all` | `propext` | `Effective/Step.lean` |
| T3 | `ledger_monotone` — the ledger is append-only along a run (I5) | induction on `ReflTransGen` | `propext` | `Effective/Ledger.lean` |
| L3.4 | `ledger_idem` (no-op re-commit) / `advance_nodup` (no duplicates) | `cases` | — / `propext`,`Quot.sound` | `Effective/Ledger.lean` |
| L3.5 | `L_not_function_of_C` / `C_not_function_of_L` — bookkeepers independent (modest direction) | witnesses | `propext` | `Effective/Ledger.lean` |

Tracks: `Keys.lean` (names / I1), `Scopes.lean` (names / I1, the string layer),
`Step.lean` (rules / I6), `Ledger.lean` (rules / I5).
Next: L1.9 (compose gather frames × scope atoms in `Key`); extend T2/T3 to
compound ops; stand up the Quint model that imports L1.5 + T2 + T3 (and Quint
Connect against the real handler).

## Build

Pinned: **Lean `v4.33.1`** (`lean-toolchain`), **Mathlib `v4.33.1`**
(`lakefile.toml`, exact deps in `lake-manifest.json`).

```sh
# one-time: install elan (the Lean toolchain manager), official source
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

lake exe cache get   # download prebuilt Mathlib oleans (skips the hours-long compile)
lake build           # builds Effective.Keys; `decide` checks the regression
```

`/.lake` (build output) is gitignored; `lake-manifest.json` is committed so the
dependency set is reproducible.
