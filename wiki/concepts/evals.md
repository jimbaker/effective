# Evaluation

An evaluation is a run of a workflow under a handler, and the run's record is the instrument. A
recorded run can be scored again, replayed as a regression test, and forked at one decision with the
prefix free, because a recorded result is served and never re-executed
([concepts/semantics](semantics.md)).

## What is built

| piece | does | where |
|---|---|---|
| `Judge` | a domain op asking bounded, typed questions about one state; served by Jev, by a scripted table in a test, or from a recorded tape | [concepts/judgment](judgment.md), [`src/effective/interpreters/jev.py`](../../src/effective/interpreters/jev.py) |
| `improve` | proposes candidates, scores each into a `Measurement` (an objective vector plus its textual diagnostic), keeps the Pareto frontier, repeats for a bounded number of rounds | [`src/effective/improve.py`](../../src/effective/improve.py) |
| `dominates`, `frontier` | Pareto dominance over objective vectors such as cost, quality and latency | [`src/effective/pareto.py`](../../src/effective/pareto.py), its laws tested in [`tests/test_pareto_laws.py`](../../tests/test_pareto_laws.py) |
| `bench_sweep` | `improve` with one round over a fixed grid of configs: best-of-n | [`src/agent/bench_sweep.py`](../../src/agent/bench_sweep.py) |
| `evaluate` | runs `run_agent` over a task suite under `MeteredInterpreter` and `DurableHandler`; quality is the pass rate, and the summed `Usage` carries cost and latency | [`src/agent/eval.py`](../../src/agent/eval.py) |
| `record_then_replay` | records a task once, then replays it with a model and tools that raise if called; `replay_is_free` reports that they never were | [`src/agent/bench.py`](../../src/agent/bench.py) |
| scorers and `rescore` | a scorer is a pure function of a recorded answer and its task, so `rescore` applies a new metric to recorded trajectories with no model call | [`src/agent/scoring.py`](../../src/agent/scoring.py), [`src/agent/bench_telemetry.py`](../../src/agent/bench_telemetry.py) |

[ADR-0005](../../docs/adr/0005-optimizer-agent-in-the-loop.md) is Accepted and is built as `improve`. [ADR-0006](../../docs/adr/0006-two-stage-marginal-pareto-selection.md), the two-stage selection rule
that would promote a candidate from forked traces, is Proposed: the fork driver and `frontier` it
composes over exist, and its screen, divergence detector, replicated confirmation and promotion rule
do not.

## The benches

| bench | asks | ground truth |
|---|---|---|
| [`src/agent/gsm8k.py`](../../src/agent/gsm8k.py) | grade-school math as a multi-turn ReAct task with calculator tools | the gold number after GSM8K's `####` marker, from a vendored sample |
| [`src/agent/hotpotqa.py`](../../src/agent/hotpotqa.py) | multi-hop QA with each paragraph a tool, so reading a distractor is a labeled trap | exact match and token F1 with the benchmark's own normalization |
| [`src/agent/skillsbench.py`](../../src/agent/skillsbench.py) | the same tasks with skills absent, inlined, or disclosed from a catalog | each task's own verifier, run in its container |
| [`src/agent/contrastbench.py`](../../src/agent/contrastbench.py) | a REPL-style loop outside the substrate against the typed, checkpointed `run_code` loop over seeded log-aggregation questions | computed by the generator; scoring consults no model |

## The record as an instrument

| instrument | answers | where |
|---|---|---|
| replay | did the workflow change? A recording is a fixed test set, and `ReplayHandler` refuses a run that yields a different op sequence | [concepts/testing](testing.md) |
| `fork_at` | what if this one decision had gone differently? It replays the trace up to op `at`, substitutes one result there, and runs the tail live under a metered domain | [`src/effective/fork.py`](../../src/effective/fork.py) |
| `run_fork`, `marginal_sweep` | the same on a durable engine; a sweep forks one base N ways as child tasks and returns their outcomes in the order asked | [`src/effective/fork.py`](../../src/effective/fork.py) |
| `Forked`, `ForkSealed` | the fork's provenance: a genesis row in the child's run naming the parent run, the fork point and the substitution, with every row of the child marked hypothetical | [`src/effective/counterfactual.py`](../../src/effective/counterfactual.py) |
| `compare`, `fork_marginal` | the difference between the two lineages, aligned on canonicalized keys | [`src/effective/lineage.py`](../../src/effective/lineage.py) |

`fork_at` substitutes only at a decision, a `Step` or an `AwaitEvent`, and refuses a sleep, a write
or a concurrent op as the fork point. A forked tail runs live, so a fork costs what its tail costs
and the prefix is free.

## The signal that steers is not the instrument that accepts

A measure an optimizer climbs stops measuring what it was chosen for, so the score `improve` uses to
rank candidates is kept apart from the check that accepts the result.

| practice | where it holds |
|---|---|
| a green `just check` gates every candidate, and a frontier claim waits for a held-out set | [ADR-0005](../../docs/adr/0005-optimizer-agent-in-the-loop.md), its invariants |
| the objective is a vector with a diagnostic beside it, which is harder to game than one self-rated number | `Measurement` in [`src/effective/improve.py`](../../src/effective/improve.py) |
| a review and its ruling are separate states, each appending its own canonical row | [`tests/test_machine_fuse_or_split.py`](../../tests/test_machine_fuse_or_split.py) |
| a coding verdict the machine can compute is a pure function of what was measured, and no model grades it | `mechanical_judges` in [`src/effective/coding/verdicts.py`](../../src/effective/coding/verdicts.py) |

## What is not built

| piece | state |
|---|---|
| a trial, grader and suite vocabulary | not built in the substrate; each bench names its own result row (`TaskResult` in `eval`, `TrialResult` in `skillsbench` and in `contrastbench`) |
| pass@k and pass^k | not built; `skillsbench` reports a pass rate with a pass as reward 1.0, the pass@1 convention for a binary verifier |
| a dataset as a selection of recorded runs with provenance | not built; a bench loads its tasks from a vendored file or a generator |
| promotion of a winner to the ledger | not built by any consumer of `improve` |
