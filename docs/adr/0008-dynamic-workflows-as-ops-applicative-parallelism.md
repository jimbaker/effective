# ADR-0008: Dynamic workflows are agent-layer ops; applicative parallelism (`gather`)

- **Date:** 2026-06-19
- **Status:** Accepted for `gather`, which is built on every handler; the agent-fleet op is
  served by spawned child tasks and has no op kind of its own. Built: the `Gather` op
  ([`src/effective/ops.py`](../../src/effective/ops.py)) and `gather` ([`src/effective/api.py`](../../src/effective/api.py)); concurrent branches on a ctx
  that advertises `concurrent_safe` (`SqliteTaskContext`, `ConcurrentAbsurdCtx`), with parallel
  tools and serialized writes; an await inside a branch, on a ctx with the `peek_event`
  capability, and a sleep inside a branch. Unbuilt: a bounded-concurrency cap, and per-branch
  connections (Option A, §3).
- **Relates to:** [ADR-0002](0002-harness-layer-stack.md) (the op and domain layer stack; branch keying reuses the deterministic
  naming of the injected permission `AwaitEvent`), [ADR-0009](0009-durable-backend-two-regimes-taskcontext.md) (the `TaskContext` the durable gather
  runs on), [ADR-0021](0021-dynamic-graphs-as-projections.md) (a spawned fan-out joins in a loop), [ADR-0025](0025-race-and-quorum.md) (`race` and `quorum`, the
  first-k siblings of `gather`).

## Context

An agent orchestration script (a fleet of subagents run by `pipeline`, `parallel` and
schema-typed agent calls) and an Effective workflow sit at different altitudes. The script is the
exploration layer: fan-out, fresh-eyes subagents, a distilled artifact at the end. Effective is
the durable execution, replay and optimization substrate. Every property the script lacks
(durability, mid-run HITL, an append-only canonical record, model-free replay, metered Pareto
optimization) is what makes a learning loop compound across runs where a transient script
discards its run when it ends.

Effective already assumes a capable agent operates its workflows (`run_agent`, the ReAct loop,
the model callers). Running a whole fleet as one op of an Effective workflow is that assumption
one level down. A fleet run is nondeterministic like any single model call, so it is recorded
once and never replayed by re-running.

Two parallel shapes recur:

| shape | example | needs |
|---|---|---|
| fan-join | N homogeneous branches, all of which must complete; an N-run ensemble | an op the handler interprets, with keys that survive any interleaving |
| monitor | a long-lived worker and an observer bound to its lifetime, watching it while it runs | no op: an inspect-only read of disposable state is telemetry |

## Decision

### 1. A fleet run is an op on the durable spine

An Effective generator is the spine; a fleet run is one recorded op whose result the handler
checkpoints. The split is by concern:

| concern | owner | mechanism |
|---|---|---|
| agent work: nondeterministic, expensive | the fleet | one op; nondeterminism sealed behind one checkpoint |
| determinism: pure gates and thresholds | the domain layer | pure ops and channels, testable in isolation |
| HITL ratification | Effective | `await_event`, durable suspend and resume |
| canonical record | Effective | the append-only ledger; rankings are derived projections |
| evaluation | Effective | Recording then Replay regression, with zero model calls |

**Granularity is the phase.** Each phase result is a recorded op; fan-out within a phase stays
the fleet's concern. The optimizer tunes what it can meter: the skill version a phase dispatches,
the model tier per phase, the pure gates' thresholds, and the run count N. It gives up tuning
sub-prompts inside a sealed phase, which is the noisiest knob and the right one to lose first.
Re-expressing a phase as native `ask_llm` ops is the migration path, and the side-by-side of the
two expressions is the demonstration that they are one topology.

**The fleet op has no op kind of its own.** It decomposes into ops that exist: a checkpointed
spawn of a child task and an `AwaitEvent` on that child's done event (`spawn_child` and
`join_child` in [`src/effective/spawning.py`](../../src/effective/spawning.py); `effective.compose.spawn_subagent_task` composes them,
and [`tests/test_spawned_subagent.py`](../../tests/test_spawned_subagent.py) proves a parent crash neither re-spawns nor restarts the
child). The handler holds a spawn to the task's depth budget and refuses one past it. An N-way
fleet is a loop of spawns followed by a loop of joins (`marginal_sweep` in
[`src/effective/fork.py`](../../src/effective/fork.py)), never a `gather`: the children already run concurrently in their own
tasks, and a branch's rescoped await would not match the child's unqualified emit. The join rides
an ordinary event, so it runs on both engines.

### 2. `gather` is an effect

`results = yield from gather([b0, b1, ...])` is a control-axis op that composes other ops. It is
never a host-language helper that starts `asyncio` tasks itself:

- **Determinism boundary.** Parallelism started by a helper happens outside the op stream, where
  no handler can see, record or replay it. As a yielded op, the independent set is data the
  handler receives.
- **Swappable interpretation.** One workflow, three meanings:

| handler | interprets `gather` as |
|---|---|
| `RecordingHandler` | structured concurrency: an `asyncio.TaskGroup`, each branch in its own child handler on a thread; results joined in branch order; branch failures joined into an `ExceptionGroup` |
| `ReplayHandler` | structure: it re-runs the branch generators in index order, with no concurrency, feeding each leaf its recorded result |
| `DurableHandler` | each branch runs through a child handler whose keys carry the branch frame; branches run concurrently on a `concurrent_safe` ctx and in index order otherwise |

`gather` is the applicative dual of `yield from`'s monadic bind. Sequential `yield from` permits
each step to depend on the last; `gather([a, b, c])` asserts that its branches do not, and that
assertion licenses both concurrent execution and order-independent replay keying. It is the
control-axis twin of combining independent channel outputs on the data axis.

**Keys are structural.** A branch's keys carry the frame `gather:{g},{i};` (`gather_frame` in
[`src/effective/api.py`](../../src/effective/api.py)): the gather's ordinal and the branch index, never completion order. The
trace records the leaves under those keys and holds no entry for the gather itself. A crash
mid-gather keeps every committed branch step and re-runs only the rest, because only successful
steps are checkpointed.

**A branch may park.** A branch's `await_event` peeks (`peek_event`) and parks as a value, and a
branch's sleep compares its wake time against the clock and parks the same way; after the barrier
the run re-arms one real wait. A ctx without `peek_event` refuses an await inside a
branch with `NotImplementedError`, and a wait that names a deadline is refused inside any branch
(`refuse_a_bounded_wait_in_a_branch`, [`src/effective/ops.py`](../../src/effective/ops.py)).

**The monitor shape is telemetry.** A worker runs as an ordinary op; an observer reads its spans,
which live outside the engine and are off by default. The observer reads the disposable
bookkeeper and the worker writes the canonical one, so watching records nothing and replay has
nothing to observe. An op carrying only the read would be a second mechanism for telemetry.

**Naming.** The join combinator mirrors orchestration vocabulary (`gather`, `parallel`), so a
script and a generator read with the same words. `fork` stays reserved for counterfactual
lineages ([ADR-0006](0006-two-stage-marginal-pareto-selection.md)).

**Scheduling is the handler's concern.** The generator declares the independent set; a cap on
live branches belongs in the handler. No cap is built.

### 3. The concurrency model: one write lock, released around the work (Option B)

A durable gather holds one `write_lock` around the connection I/O and runs each step's thunk (the
tool or model call) outside it. The SQLite engine shares the lock between ctx and ledger
([`src/effective/sqlite.py`](../../src/effective/sqlite.py)); `ConcurrentAbsurdCtx` ([`src/effective/handlers/absurd.py`](../../src/effective/handlers/absurd.py)) does the
same over the Absurd SDK's public `begin_step`/`complete_step` split, so branch tools overlap
while commits serialize on the one task connection. `ConcurrentAbsurdCtx` is constructed only in
tests; a worker passing the raw SDK ctx runs branches in index order.

The shape is CPython's GIL:

| GIL | durable `gather` |
|---|---|
| one interpreter lock | one `write_lock` per task connection |
| released during blocking I/O | released during the thunk |
| I/O-bound threads overlap | tool-bound branches overlap |
| CPU-bound threads serialize | write-bound branches serialize |
| escape: free threading | escape: per-branch connections under MVCC (Option A) |

It is the right default for the GIL's own reasons: model-latency-bound fan-out holds the lock for
a vanishing fraction of the wall clock, so nothing pays for parallelism it does not use.

**The escape is safe by construction.** The GIL is hard to remove because objects are shared and
mutated. Gather branches share nothing: independence is the applicative precondition, the
structural keys make checkpoint state disjoint, and the ledger is append-only. Option A is
therefore safe whenever writes become the bottleneck, and ownership machinery such as cowns buys
nothing, having no sharing to manage. The lock stays because writes are not the bottleneck.

**The partial order is the interleaving semantics.** One `ctx.step` is atomic; a branch's
sequence of steps is not serialized against another branch's, and per-branch order is all the
ledger promises. Tests assert branch-ordered results exactly and cross-branch ledger order as a
set; crash tests target a fault by step name, which is well defined under any interleaving. The
critical section is explicit and tunable: it could shrink to `complete_step` alone, or drop to
Option A per resource.

### 4. Many runs under nondeterminism

One fleet run is one sample. `gather` makes N runs a fan-join:

- **Vote.** A finding in k of N runs is high-confidence; a finding in 1 of N is a variance artifact
  to flag, never to drop silently.
- **Variance is a Pareto axis.** High spread across runs marks a genuinely ambiguous case to route
  to a human. At equal mean, a low-variance candidate dominates a high-variance one on
  reproducibility.
- **Effective's part.** N recorded ops, all N in the ledger as evidence, the vote and variance as
  a projection, and a replay that re-derives the vote without running anything.

## Consequences

- A one-off capture procedure becomes a replayable workflow, and a model swap is checked by replay
  against a frozen run rather than asserted.
- The determinism boundary is unchanged: fan-join keys by structural position, and a watcher reads
  telemetry.
- The combinators are substrate; domain gates and rules stay with the domain.

## Open questions

- How a handler-side scheduler interacts with checkpoint granularity on a new backend.
- A write model finer than one lock: the keys already prove branches write disjoint state, so
  per-branch connections would let disjoint commits run in parallel at the multi-writer tier, and
  would buy little on single-writer SQLite.
