# ADR-0018: Measured-spend accrual: usage in the checkpoint for the in-task trip, the ledger for a cross-task pool, and a confluent per-branch fold with no lock

- **Date:** 2026-07-17
- **Status:** Accepted. Built: the `{result, usage}` checkpoint envelope behind the spawn-param
  contract (`CONTRACT_PARAM`, `Contract` in [`src/effective/cost.py`](../../src/effective/cost.py)), the handler-owned
  replay-derived meter with per-branch subtotals folded at the gather barrier, the sequential
  trip and park (`MeasuredBudget`, `enforce_measured` in [`src/effective/budget.py`](../../src/effective/budget.py), driven by
  `DurableHandler._enforce_measured`), the accrual carry across a `respawn`, conformance on both
  engines. Unbuilt: the cross-task pool (§6), the ledger half of the dual write, per-branch
  sub-budgets inside a `gather`, the per-generation ceiling's enforcement (§5), and carrying a
  `govern` gate's grants across a `respawn` (§8).
- **Relates to:** [ADR-0002](0002-harness-layer-stack.md) (harness layers: `metered` is a `@domain_layer`), [ADR-0008](0008-dynamic-workflows-as-ops-applicative-parallelism.md) (applicative
  `gather` and structural checkpoint keying), [ADR-0009](0009-durable-backend-two-regimes-taskcontext.md) (one `TaskContext`, no SQLite-specific
  handler), [ADR-0016](0016-formalization-and-operational-semantics.md) (the serialization-injectivity family), [ADR-0019](0019-govern-serve-composition-combinators.md) (the measured trip as a
  `govern` policy, `as_policy`).

## Context

The structural budget dimensions (depth, breadth) are deterministic threaded counters with no
handler-side state. The measured dimensions (dollars, tokens) are different: their trip depends on
nondeterministic spend reported by the provider.

**The meter is empty after replay.** Spend is accrued by `metered`, a `@domain_layer`, inside the
checkpoint thunk. Replay returns the committed value without running the thunk, and the checkpoint
stores the bare result, so the meter reads zero after any replayed prefix and the layer can never
re-trip. The trip has to become a function of recorded state.

Three mechanisms were candidates:

| mechanism | idea |
|---|---|
| (i) usage in the checkpoint | record usage beside the result, so replay re-folds it and the trip re-derives at the same op |
| (ii) a recorded budget-poll op | a checkpointed poll at each gated boundary |
| (iii) ledger-projected spend | fold spend from the canonical ledger |

They are not parallel alternatives. (ii) records the trip verdict rather than the spend: a poll
after a resume runs its thunk fresh and reads the same empty meter, so (ii) as a trip source
re-imports the bug. (iii) as the in-task trip source has a replay-prefix problem: on replay the
ledger already holds the whole run's rows, so a fold at op 3 would see op 9's spend, and bounding
it to the replayed prefix means correlating rows to op positions, which the checkpoint stream
already does. So (iii) as a trip is (i) with extra steps. But (iii) is the only mechanism for a
pool shared across tasks, since a Python object cannot cross a `spawn` to another worker. The real
axis is where spend becomes recorded state, and the two stores play different roles.

## Decision

### 1. Roles: (i) for the in-task trip, (iii) for a cross-task pool and audit, (ii) dropped

| job | mechanism | why |
|---|---|---|
| in-task measured trip (the park gate) | (i) usage in the checkpoint | re-derives at the same op on replay; the checkpoint stream already correlates spend to op positions |
| cross-task spend pool and canonical audit | (iii) ledger-projected spend | the pool must live in a shared store, and spend is a business fact whose canonical home is the ledger |
| observables such as wall-clock | the recorded-poll pattern of (ii) | a poll records an observation; it cannot source a spend trip |

Writing the provider's usage to both the checkpoint and the ledger is a dual write of one fact,
which the bookkeeper invariant permits; computing either book from the other stays forbidden.

### 2. The envelope, and a versioned replay contract

Accrual moves above the checkpoint boundary by enveloping the model call's `Step` checkpoint value
as `{result, usage}` on the same row. A separate `usage:{op_key}` row is broken by a crash between
the two commits: the first step replays without its thunk, and the usage lived only in the provider
response.

This is a checkpoint-schema change, and sniffing for a dict with `result` and `usage` keys would
collide with a legitimate workflow result of that shape. So the replay contract is versioned
explicitly and fixed at spawn, in the task's immutable spawn params: `CONTRACT_PARAM` is a reserved
key, read by `Contract.from_params`. Absent means `Contract.V0` (bare results); `"v1"` means the
enveloped path. A spawn param is present or absent from birth, immutable and replayed identically,
so a v1-capable worker resuming a task spawned before the envelope existed reads no param, stays
on v0, and keeps writing bare rows; no task ever holds mixed rows. A start-of-task checkpoint
marker would not work: it is absent for a fresh v1 task and for a resumed v0 task alike until the
first step writes it. Pinned by `test_measured_v0_task_stays_v0_under_a_v1_capable_worker`
([`tests/test_conformance.py`](../../tests/test_conformance.py)) and `test_v1_handler_on_a_v0_checkpoint_fails_loud_not_silent`
([`tests/test_budget_accrual.py`](../../tests/test_budget_accrual.py)).

### 3. A handler-owned, replay-derived meter

`metered` cannot host the enforcement trip: it is a `@domain_layer`, which may not yield an
`AwaitEvent`, and it sits below the checkpoint. The fold moves to op grain, above `ctx.step`,
owned by `DurableHandler`.

- The handler drives the domain's `run_metered`, which returns `(result, usage)` without
  `metered`'s unwrap. The handler writes the envelope, unwraps above the checkpoint, and the
  workflow still receives the bare result, so workflow code is unchanged.
- The replay-derived meter is exposed read-only to op layers and grantors (`current_meter`): a
  park reads recorded state, never a live side channel.
- `MeteredInterpreter.meter` is live telemetry and not an enforcement bookkeeper. Kept as
  telemetry, it is constructed with `budget=None`, since a `CostBudget` makes its pre-forward
  `exceeded()` refusal a second enforcement point racing the handler's trip. In the tree the
  budget-bearing constructions are agent benches ([`src/agent/eval.py`](../../src/agent/eval.py),
  [`src/agent/contrastbench.py`](../../src/agent/contrastbench.py), [`src/agent/skillsbench.py`](../../src/agent/skillsbench.py), [`src/agent/debug.py`](../../src/agent/debug.py)), which run
  in-process and are the case this permits.

### 4. Concurrency: a per-branch fold at barriers, no lock

| design | verdict | evidence |
|---|---|---|
| per-branch subtotal, folded at the gather barrier in branch-index order | confluent on every schedule, lock-free by construction | Lean `Effective.Budget.operational_run_eq_executedB` (a conservation law) and its corollary `operational_confluent`; Quint `budget_confluence.qnt` `confluent` |
| one shared meter gated mid-branch | the trip fires at a schedule-dependent op, so replay diverges | Lean `shared_gate_order_dependent`; the `sharedMeter = true` tooth in [`scripts/formal_checks.sh`](../../scripts/formal_checks.sh) |
| a shared meter behind a lock | still non-confluent: a lock cures lost updates, not the trip's schedule dependence | Lean `shared_lock_insufficient`; the Quint model is atomic and still violates confluence |

Each branch handler accrues into its own subtotal, single-threaded within the branch, and the
parent sums them at the barrier. For enforcement in the middle of a fan-out the design is
per-branch sub-budgets, deterministic within each branch's own op order and the same shape as a
successive-halving allocation; that partition is unbuilt. The locks on `CostBudget.add` and
`MeteredInterpreter._accrue` keep the telemetry totals correct under concurrent accrual and play
no part in enforcement.

### 5. The trip, the park, and the grant name

`MeasuredBudget(run_id=..., overall=..., per_generation=..., on_exhaust=...)` is the measured
ceiling. At a sequential program point, before a v1 model call, the handler classifies with
`enforce_measured`:

| outcome | when | effect |
|---|---|---|
| `Cleared` | spend is under `overall + Σ grants` | the call runs |
| `Parked` | over the ceiling, `on_exhaust="park"` (the default), no answer delivered for this trip | parks on `budget-grant:{run_id},{trip}` |
| `Exceeded` | over the ceiling with `on_exhaust="fail"`, or the grant answers `stop` or a non-positive amount | raises `BudgetRefused` |

A positive `Grant.add_dollars` refills and re-checks. `trip` counts recorded prior trips, so it is
deterministic and a grant name never answers a later trip. The run id is in the name because
durable event names are global. `on_exhaust="fail"` serves an unattended batch.

The trip is a pre-check: it refuses or parks the next call once spend has crossed the ceiling, and
a call's cost is unknown until it runs. So the bound is
**`spend ≤ overall + Σ grants + one in-flight call`**, bounded and non-compounding. Lean states it
as `run_bounded_by_limit_grants_and_one_call` with `run_overshoot_is_tight` in
`EnforceMeasured.lean`; the conformance pin is
`test_measured_trip_just_sufficient_grant_respects_the_overshoot_bound`.

Measured spend is gated at sequential points only. Two branches of a `gather` tripping in one
round would compute the same `trip` from the same recorded state and so the same grant name, so a
branch handler carries no `MeasuredBudget`. Branch-level measured parks would need the branch
frame in the grant name.

Across a `respawn` chain the two ceilings ask different questions: `overall` caps the run across
every generation, and `per_generation` is an allowance that re-arms. The handler carries its
accrual `(spent, granted, trips)` to the successor in a reserved spawn param (`ACCRUAL_PARAM`), so
`overall` holds across generations. `enforce_generation`, the per-generation transition on
`generation-grant:{run_id},{generation},{trip}`, is written and driven by no handler; pinned as a
strict xfail, [`tests/test_respawn_durable.py::test_a_per_generation_ceiling_is_ACTUALLY_ENFORCED`](../../tests/test_respawn_durable.py).

### 6. The cross-task pool: a shared store, serialized by the database

A budget shared across a durable subagent tree is append-only usage rows and a projected spend
total. A gate at spawn grain reserves or consumes against the projection, coarse by nature, so its
cost scales with the spawn rate. Serialization is a database transaction (an atomic reserve or
`UPDATE … SET spent = spent + :u`), since cross-task state must live in a shared store anyway.
Whether the substrate is a budget row with atomic reserve and consume or a ledger-projected fold
is decided when the pool is built.

### 7. One change, both engines

There is one durable handler over the `TaskContext` protocol ([ADR-0009](0009-durable-backend-two-regimes-taskcontext.md)), so the envelope and the
fold land once, in `DurableHandler` and its serde, and [`tests/_conformance.py`](../../tests/_conformance.py) carries the gates
to both engines. The pins in [`tests/test_conformance.py`](../../tests/test_conformance.py) cover park, grant and resume
(`test_measured_trip_parks_grants_and_resumes`), worker death
(`test_measured_trip_survives_worker_death`), the overshoot bound, `trip` re-derived across
replays with several grants, and a concurrent runaway inside a `gather`
(`test_measured_concurrent_gather_runaway_trips_deterministically`, with its worker-death twin).

### 8. A `govern` gate's grants and a `respawn`

A `govern` budget gate ([ADR-0019](0019-govern-serve-composition-combinators.md)) keeps its answers in a per-attempt run scope, re-derived from
the event store on replay, and a `respawn` successor starts that scope empty. The handler's own
accrual carries (§5), and the gate's answers do not, so a gate-driven successor parks again under
a ceiling its predecessor's grant raised. The decision is to carry the gate's **answer history**
across the respawn beside the spend, with `govern` owning a read of its run-scoped answers and a
seed for the successor's; `BudgetPolicy`'s fold is unchanged and sees every generation's answers,
and a `stop` carries. Rejected: a cell the policy writes back to (a policy that writes stops being
a function of its `GateState`), and a recompute at respawn (the handler would have to know which
layers drive a budget). Unbuilt.

## The `at` index convention

The measured machinery indexes ops at several sites in two coordinate systems: all ops (every
yielded op, what "fork at the review decision" means) and Step-only (what the free-replay prefix
counts). The public `at` is all-ops, `OpIndex`; the Step-only `StepIndex` is internal; and one
named conversion, `to_step_index` in [`src/effective/fork.py`](../../src/effective/fork.py), crosses at the one boundary. Both are
`NewType`s, so `ty` rejects one where the other is wanted. The conversion's agreement with the
bridge's own Step filter is pinned in [`tests/test_at_index.py`](../../tests/test_at_index.py). The theorems here are stated over
concern-local sub-streams (the trip index, the cost list), so the conversion is a precondition
below them; a Lean boundary law waits until the crossing has more than one site.

## Alternatives rejected

| alternative | why it lost |
|---|---|
| (ii), a recorded budget poll, as the in-task trip source | a post-resume poll reads the empty meter |
| (iii), ledger-projected spend, as the in-task trip source | bounding the fold to the replayed prefix reinvents the checkpoint stream's op-position correlation, plus a query per gate |
| a lock on a shared meter | fixes the total, not the trip; formally non-confluent even when atomic |
| usage in a separate `usage:{op_key}` checkpoint row | broken by a crash between the result commit and the usage commit |
| a start-of-task checkpoint marker for the contract version | cannot distinguish a fresh v1 task from a resumed v0 task at the first step |

## Consequences

- The measured park and ask is available: a run that crosses its ceiling waits durably for a
  grant, resumable from recorded state, with per-seam spend attribution, and a fork can re-run the
  free recorded prefix and probe both grant arms before the grant is decided.
- Enabling it is a spawn-side choice: set `CONTRACT_PARAM` to `"v1"` and pass a `MeasuredBudget`.
  Tasks spawned without the param replay on v0.
- `CallTool` cost is out of scope until a tool reports usage; only model-call `Usage` is
  enveloped. Tokens ride the same `Usage`.
- Bench cost totals read the handler-owned meter or a ledger projection; `MeteredInterpreter.meter`
  is telemetry.
