"""The coding machine: a coding agent as a STATE MACHINE over the effect substrate.

Two grains, composed — and which layer owns what is the whole design:

    convergence loop ──▶ PLAN ──▶ [ worker ──▶ judge ──▶ transition ] ──▶ postamble
                                  └─ within a state ─┘   └─ between states ─┘

That diagram is THIS embodiment's shape, not the substrate's. The `worker -> judge` pair inside
a state is `machine.specs.fuse`'s doing, a construction an embodiment may decline; the
interpreter itself runs ONE slot per state. A judgment that needs an address of
its own (a budget line, an edge in the table, a name a pack can bind to) is written as a state
instead; `tests/test_machine_fuse_or_split.py` carries both forms in one machine. Every state
here fuses, which is the right default and not a limit.

**Between states, a judge supplies a VALUE and a pure total function picks the next state.**
Each state has its own verdict enum, so the transition's domain is the dependent sum
Σ_s Verdict(s) rather than the product State x Verdict — a product would demand an arm for
`(TEST, ReviewVerdict.APPROVED)`, which no judge can produce. `assert_never` closes each
per-state router, and that is the gate: `ty` narrows an enum match to `Never` (measured
2026-08-15), so a dropped or added arm is a **type error**. `--totality src` sees none of it —
its census emits zero rows for a `match` over an enum — so a green lint is not evidence here.

**Within a state, the model chooses among a repertoire-masked set of ops**, which is not a
control-flow violation but the measured design. That is `run_agent` one grain down, and it is
why the two designs this package composes were kept whole rather than converted into each other.

**Exhaustion is minted by the INTERPRETER, never by the judge.** The trampoline's final `descend`
level injects `Exhausted` *instead of consulting the judge*, as `budget.OnExhaust="park"` does.
`Exhausted` is deliberately outside every verdict enum so a judge's `Effect[V]` cannot spell it; a
judge-supplied exhaustion would relocate the livelock rather than kill it, because a judge that
never says it loops forever.

**The commitment postamble runs on EVERY exit path** — approved, exhausted, or refused. It commits
what exists, runs the predicate, and records the verdict whether it passed or failed, which is the
property that makes the livelock impossible by construction rather than by discipline. The
incumbent
(`tests/_coding.py`) tested its outcome nominally (`outcome != "pass"`) and appended 99 canonical
rows in 400 ops on a case-variant typo; this package exists to make that unrepresentable.

Two bookkeepers, and a state DECLARES which it reaches: `StateSpec.canonical` says whether a state
touches the append-only ledger, rather than leaving it to be discovered two call frames away.

Keys carry two scopes, and dropping either one loses a distinction. `state:{name}` names which
code runs. `d:{n}`, the trampoline's level, counts re-executions of one position: a re-entered
`state:` frame gets no occurrence suffix, so `TEST → DRAFT → TEST` under `state:` alone mints two
identical keys on the tape, and every reader that derives identity from position (`graphview`, the
key registry, the paths sweep's injectivity invariant) loses the second visit. The durable engines
number repeated steps `name#N` within an attempt, and `await_event` has no such rule, so two awaits
of one name bind the same payload. Backedges are this machine's reason to exist, so every key
carries its level.

`coding_machine_wf` (`tests/_conformance.py`) drives this package through `DurableHandler` on both
engines, and `tests/test_conformance.py` pins the walk, the content-addressed artifact id, the
postamble's two ledger rows and the keys as identical across them, and again after a crash at
every op boundary.

This package imports `effective` and never `agent`, which is why the ReAct loop lives in
`effective` rather than being imported from `agent`.
"""
