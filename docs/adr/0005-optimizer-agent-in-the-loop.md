# ADR-0005: The optimizer is a coding agent in a loop: legible GEPA over reasonable workflows, staged to bandits

- **Date:** 2026-06-16
- **Status:** Accepted. The loop is built as the `improve` combinator (`src/effective/improve.py`),
  with three consumers: `debug_loop` (`src/agent/debug.py`), `bench_sweep`
  (`src/agent/bench_sweep.py`) and `tune_permits` (`src/agent/permit_tuning.py`). Fork-as-replay
  is built (`fork_at`, `src/effective/fork.py`) and serves value-of-information probes; wiring it
  in as `improve`'s marginal move is not. Not built: a typed `ForkDelta` candidate, bandit
  allocation, the legibility ratchet, and a ledger promotion of a winner by any consumer.

## Context

ADR-0001 deferred categorical strategy seams until an optimizer exists, picturing the optimizer
as a separate subsystem to be built later. That framing is too narrow. GEPA (reflective prompt
evolution that selects on a Pareto frontier and learns from natural-language traces) has four
ingredients, and the substrate already supplies three:

| GEPA ingredient | Effective substrate |
|---|---|
| optimizable components | typed channels and `Prompt.seams` (ADR-0001); config dimensions such as catalog rendering, retrieval `k`, gate thresholds, model choice |
| rollouts and multi-objective fitness | `BenchSpec`, `Scorer`, `frontier_points`, scoring as replay (ADR-0004 D6) |
| reflective natural-language feedback | t-string telemetry: which ops fired, cost, latency, which gate tripped, which repair looped, each span carrying `code.*` call-site attributes |
| the reflective mutator and selector | the `improve` loop: `propose` reflects, `score` measures, `effective.pareto.frontier` selects |

The fourth ingredient, the reflective mutator, is often a coding agent. A coding agent reading a
legible workflow sees more than a prompt optimizer's reflective model, which reads one prompt
string in isolation: it reads the whole workflow. So the optimizer is staged rather than built
whole, and its first stage is the legible one.

## Decision

Optimization is a first-class capability whose reflective mutator is, by default, a coding agent
reading a legible workflow. It is staged:

| stage | the mutator | the selector |
|---|---|---|
| 1 agent in the loop | a coding agent proposes a type-checked edit to a seam, a channel or a config dimension | the frontier, re-benched on replay |
| 2 fork-mechanized marginals | the agent proposes which marginal move to try; the harness executes it as a scored counterfactual | the frontier over forked tails |
| 3 automated | GEPA-style reflective generation of new candidates | sequential-bandit allocation of the rollout budget |

Two principles follow:

- **Legibility is the optimizer's interface.** The static surface built for human maintainers
  and for `ty` and the linters is the optimizer's mutation API. An illegible workflow is an
  unoptimizable one.
- **Scope by typing.** A `Gated` or `FormGate` channel and the signature validator bound what a
  mutation may touch inside a seam, and a green `just check` is a hard constraint on an accepted
  candidate. The guardrail that protects the workflow from the model protects it from the model's
  optimizer too, which is the defense against the illegible local optima raw prompt optimizers
  overfit into.

## The loop as built

`improve(seed, propose, score, *, objectives, rounds, done, rubric, compact, summarize)` returns
the Pareto frontier. The stages are policies plugged into this one loop: `propose` and `score`
are parameters.

| part | holds |
|---|---|
| durability | every `propose` and `score` is a sealed op inside a `scoped` frame (`seed`, `gen:{r}`, `cand:{r},{i}`), so a crash mid-round resumes without re-paying scored candidates; scores run in parallel through `gather` |
| selection | `effective.pareto.frontier` over recorded measures: pure, so replay re-derives it with no model calls. One objective collapses the frontier to the argmax set |
| ASI | `score` returns a `Measurement(measures, asi)`, where `asi` is the actionable side-information (a test failure, a lint or `ty` message, a rubric note). `propose` reads the frontier with its ASI and reflects before mutating |
| compaction with the rubric pinned | a pure `compact` trigger fires a recorded `summarize` op over the ASI history; the `rubric` is never passed to `summarize`, so the objective's definition survives every compaction |
| termination | `rounds` bounds the search; `done` is a pure early stop, which also expresses a staged objective (correctness first, then quality) |

The three consumers show the shape generalizes:

| consumer | candidate | score |
|---|---|---|
| `debug_loop` | a code edit through the `Gated` precise-edit channel | tests, then quality against a rubric (`agent.code_quality`: ruff and `ty`) |
| `bench_sweep` | an agent config | a cost, quality and latency frontier; best-of-n is `rounds=1` |
| `tune_permits` | a `PermitPolicy` allow-table | offline replay of a recorded `(op key, human verdict)` corpus: counterfactual policy evaluation with no live risk and no model calls |

`improve` reaches a seam only when the seam's tunable is data: something that checkpoints, diffs
and mutates. A bare `Callable` (a `rules(policy)` predicate, `Gated.predicate`) is out of reach.
ADR-0017 rules that tunable seams take data and closures are for fixed policy;
`allow_table(PermitPolicy)` is its first data twin. ADR-0006 refines the selection rule.

## The later stages

**Stage 2.** A marginal move becomes a sandboxed, scored counterfactual: `fork(run, at, delta) =
replay(trace[:at]) then interpret(tail under delta)`. The replay of the prefix is free, and only
the divergent tail is paid for. `fork_at` and `replay_prefix` exist; the typed candidate (a
`ForkDelta`) and its use as an `improve` proposal do not. The same fork already serves
value-of-information probes (`src/agent/voi.py`) and counterfactual audit
(`src/effective/counterfactual.py`), so optimization would be its third use.

**Stage 3.** When marginal moves are regular enough to enumerate, the agent becomes a subroutine:

- **Generation (GEPA).** Reflective mutation proposes new candidates from the traces of
  survivors. Categorical seams (ADR-0001) earn their place here, as an enumerable typed space.
- **Selection under a rollout budget (sequential bandits).** Rollouts are the expensive resource,
  and the config-by-instance space is too large to evaluate exhaustively, so this is
  best-arm identification rather than regret minimization:

| family | role |
|---|---|
| successive halving, Hyperband | spread a small budget, drop the worse half, repeat; the first bandit to wire in |
| UCB, Thompson sampling | which candidate gets the next rollout as evidence accumulates |
| Pareto best-arm identification, racing | identify the Pareto-optimal set, matching a frontier objective |
| contextual bandits | choose the config as a function of the input: a learned router |

Allocation defaults to offline, held-out, replay-cheap rollouts, which keeps determinism clean and
spend bounded. Online exploration on live traffic spends real money per exploratory arm and can
reach the consequence boundary, so an exploratory arm there passes the permission cascade as an
`Escalate`.

## Consequences

**Positive**

- Optimization is available at stage 1 with no new subsystem.
- The objective is a cost-efficiency frontier, harder to game than a scalar reward.
- Per-seam credit assignment is sound because composition is applicative: no composed node reads
  another's resolved value (`IndependenceError`, ADR-0001), so independent seams can be attributed
  and mutated independently.

**Risks and their mitigations**

| risk | mitigation |
|---|---|
| the search is nondeterministic | reification: the search trajectory is recorded nowhere canonical; `improve` keeps the pool in memory and re-derives it on replay, and the only canonical event a caller appends is the winner's promotion |
| overfitting a small eval | held-out sets before a frontier claim; a multi-objective frontier is harder to overfit than a scalar |
| online bandits spend money and touch irreversible I/O | offline first; exploratory arms at a consequence boundary pass the permission cascade |
| an optimizer degrading legibility | a green `just check` (ruff, the lints, `ty`, the channel validator, volatile-last) gates every candidate |

## Invariants

- **Reification.** The ledger records the chosen configuration, never the search trajectory.
- **Scope by typing.** A mutation acts only within typed seams, and `just check` gates acceptance.
- **Held-out before frontier claims.**
- **Gated exploration.** An exploratory arm at a consequence boundary is an `Escalate`.
- **Placement.** `improve` and `effective.pareto` yield ops, so they are substrate in `effective`;
  the benches and consumers that score them live in `agent`.

## Alternatives rejected

- **An automated GEPA engine first.** Stage 1 is available and is the legible default; building
  the engine first inverts the value order.
- **A scalar RL reward.** Natural-language feedback is more sample-efficient than a scalar, the
  telemetry already supplies it, and the objective is a frontier.
- **Online-first bandits.** Each exploratory arm is real spend and a consequence-boundary risk.
- **The optimizer inside the runtime handler.** It would put a nondeterministic search in the
  deterministic interpretation core.

## Measuring legibility

If legibility is to ride the frontier beside cost and quality, it has to be measured, and not as
a scalar. Legibility has internal trade-offs: a terse workflow is clear to experts and opaque to
newcomers, and an over-explicit one is clear line by line and opaque as a whole. An average
reports a middling score for a workflow that is clear locally and incomprehensible globally. A
self-rated scalar is also the easiest objective for an optimizer to game; a behavioral measure
observes a capability and cannot be faked.

The instrument already exists: the optimizer is an agent whose reflective telemetry is recorded.
The shape and cost of an agent's path to a correct edit (six files read, two backtracks to locate
the seam behind a failure) is the illegibility, pinned to the lines that caused it. Improving
legibility and improving optimizability are the same gradient.

| measure | what it reports |
|---|---|
| behavioral reconstruction probes | pass or fail per probe kind (predict the next op, explain a gate trip, locate the seam behind a failure, place a guardrail); the failing kind localizes the defect |
| reader variance | N independent reads of "what would changing this seam do"; disagreement names the ambiguous seam |
| a dimensioned rubric, never summed | locality, namedness, contract completeness, surprise (code that compiles and misleads) |

The type, lint and channel-validator gates are a binary floor every candidate clears; the
measured profile is the continuous layer above it.

## Open questions

- Is a contextual bandit's router a data-axis channel or a control-axis `decide` seam
  (ADR-0004)?
- Which multi-objective best-arm algorithm fits the rollout-cost profile of metered model runs?
- How large a held-out frontier shift justifies promoting a config?
- Do shared cached prefixes couple composed seams in practice, despite applicative independence?
- Which probe battery generalizes beyond one workflow, and when does a legibility profile ride
  the shipped frontier rather than only inform the next edit?
