# ADR-0004: The ReAct loop as a durable driver over a typed step

- **Date:** 2026-06-14
- **Status:** Accepted; D4's stop of an op in flight is ruled by ADR-0026. Built: `run_agent` in
  `src/effective/react.py` with its policy seams, the interrupt race (`src/effective/interrupts.py`),
  refusal-as-observation, three subagent forms (`src/agent/runtime.py`, `src/agent/compose.py`),
  the record-then-replay bench (`src/agent/bench.py`, `src/agent/bench_telemetry.py`) and the tool
  bracketing strategies (`src/agent/bracket.py`). Not built: an RLM module in the `decide` slot, a
  cross-framework harness ablation.

## Context

The loop had to support five things at once:

| requirement | what it means here |
|---|---|
| subagents | tool selection and calling without the parent's full context |
| KV-cache performance | tools bracketed per turn without busting the cached prefix |
| RLM compatibility | a DSPy-style module or a combinator policy as the reasoning step |
| agent skills | named fragments disclosed on trigger |
| algebraic effects | HITL through an Ask tool; interruption by Esc and slash commands |

and a bench that uses replay to verify the loop works and is cost-effective.

Almost every primitive already existed: `run_agent` yielded `ask_llm` and `call_tool` ops, the
channel processor parsed typed turns with a `Gated` guardrail, `cost.py` metered `AskLLM` with
`cache_read_input_tokens`, the permission cascade suspended on `AwaitEvent`, and
`effective.pareto` scored a frontier. The work was deciding what stays invariant in the loop and
what becomes a tunable seam. The loop is one more use of Effective, so it composes with ops,
layers, channels and the permission cascade rather than introducing a parallel mechanism.

## Decision

### D1: the loop is a durable driver; the policy is a parameter

`run_agent(prompt, max_iters, *, decide, act, guard, interrupt, compact, grantor)` is a `descend`
judge with one turn per level. Every model turn is an `AskLLM` step `react:turn` and every action
a `CallTool` step `tool:{name}`, inside the level's `d:{i}` frame, so a recorded trajectory
replays with no model. The seams:

| seam | type | default |
|---|---|---|
| `decide` | `(messages, level) -> Effect[AssistantTurn]`; the RLM slot | `default_decide`, one `ask_llm` |
| `act` | `(ToolRequest, level) -> Effect[ToolResult]` | `default_act`, an opaque `call_tool` |
| `guard` | judges an action before it runs: `Proceed`, `Refuse` (the observation replaces the action) or `Ask` (parks on `guard:ask`) | none |
| `interrupt` | the race seam (D4) | none |
| `compact` | a pure trigger over the transcript; when it fires the loop records one `react:compact` step whose `TrajectorySummary` replaces the prefix, keeping the task and the last `KEEP_TAIL` messages | none |
| `grantor` | adds turns before the final level | none |

`decide` need not call a model: a pure-Python policy is a valid step and leaves no op in the
trace (`test_loop_drives_a_pure_python_step`). Four of the five requirements become what plugs
into a slot, and the loop shrinks to the trampoline that threads observations and ends the run.

### D2: RLM compatibility means the channel is the DSPy signature

A DSPy module is signature, forward pass and optimizer. A `Prompt[S]` channel template is a
signature (inputs render in, `Field`/`Gated` outputs parse out), so `Predict(signature)`
corresponds to `render(template, output=S)` and DSPy's ReAct to `run_agent`. An optimizer tunes
the step's template and the loop stays invariant (ADR-0005). Because `decide` is a value, turns
compose as combinators: the channel call is the applicative unit and `yield from` the bind. The
slot accepts an RLM module; none is built.

### D3: a subagent composes over existing ops

A subagent runs a nested `run_agent` with its own narrow tool subset. It needs no new op, and
three forms cover the design space:

| form | where | to the parent | on a parent crash |
|---|---|---|---|
| `subagent_runner` | `src/agent/runtime.py` | one opaque `CallTool` step; replay binds the child's answer from the checkpoint | the child re-runs from scratch |
| `spawn_subagent` | `src/agent/compose.py` | the child's ops inline under the parent's handler, namespaced by `scoped` | the child resumes from its checkpoints |
| `spawn_subagent_task` | `src/agent/compose.py` | a `spawn` step and an `AwaitEvent` on the child's completion; the child is its own durable task | the child is neither restarted nor re-spawned |

Context isolation is also replay isolation and the cache strategy (D5): the child carries the
bulky tool schemas on its own stable prefix, and the parent's context stays lean. An opaque
child's spend is invisible to the parent's meter; it emits spans tagged with its `agent_name`,
and an optional `accrue` folds its total into a parent meter.

### D4: interrupt is a race, recorded

Esc and slash commands are a race between a keypress and an op. The loop polls a non-blocking
interrupt channel at the phases it subscribes to: before the decide, after it, and after the
action. Each poll is a recorded `CallTool` keyed `tool:interrupt,{phase}`, so the race's winner is
a checkpoint and replay re-derives it.

| signal | effect |
|---|---|
| `Quiet` | the loop continues |
| `Redirect` | the text is the next observation; one that lands before the action runs drops it |
| `Escape` | drops the pending action and ends the run with `stop_reason="escaped"` |

A pending interrupt polled before the decide wins before the model is called. A human Ask
suspends (`await_event`); an interrupt poll must not block, which is the one distinction between
them. An op already running when Esc arrives is stopped and completes with the cancel as its
recorded result (ADR-0026); the loop ends the run with the partial output as the observation.

**Refusal is an observation.** Handlers deliver a `Refused` into the workflow with `gen.throw`,
and the loop catches it around `act`, so a denial becomes the next observation (`[denied] ...`)
and the model can route around it. An uncaught refusal propagates.

### D5: tool bracketing masks selection and keeps the prefix

KV cache wants a stable, append-only prefix; enabling and disabling tools per turn by rewriting
the catalog busts the cache every turn. The tool definitions stay stable, and the per-turn
allowlist constrains selection: a grammar mask at decode time on a local server, or a `Gated`
repair on a cloud provider. `src/agent/bracket.py` holds three strategies over one allowlist:

| strategy | catalog in the prefix | allowlist enforced by |
|---|---|---|
| `FULL` | full | nothing |
| `MASK` | full, stable | a GBNF decode mask over the `tool_name` enum |
| `MUTATE` | rewritten to the allowlist each turn | the same mask |

Measured on GSM8K-as-ReAct over a local llama.cpp server (Qwen2.5-7B, Q4, 2026-06-14): `MASK`
held a `cache_hit_ratio` of about 0.90, equal to `FULL`, and `MUTATE` about 0.72, with zero
out-of-allowlist selections. On long-context HotpotQA-distractor, `FULL` was both best quality
and cheapest, because its stable prefix earned cache that the bracketing strategies forfeited.

### D6: the bench records once and replays free

`record_then_replay` runs a task twice: under `RecordingCtx`, where steps run, the meter accrues
the true cost and the model is paid once; then under `ReplayCtx`, where each step returns its
logged result. The replay interpreter raises if a model or a tool is ever called, so a clean run
is the proof that replay costs nothing and that an opaque subagent inside a recorded step does
not re-run. `src/agent/bench_telemetry.py` extends this to evaluation: `BenchSpec` binds a suite,
`run_strategy` records each trajectory under a hard `CostBudget`, `rescore` and `replay_free`
re-measure with any metric at no model cost, and `frontier_points` maps cost, quality, latency
and cache onto a Pareto frontier.

### D7: skills compose through the channel processor

A skill (name and description always present, body loaded on trigger) fits D5's cache story:
bodies load as the volatile tail, so everything before stays cache-warm. A `Template` prompt to
`run_agent` renders through `effective.channels` with no registry, so a `Skill` node in it must
carry a pin; an unpinned one is a located `SkillResolutionError` rather than a body read from
whatever registry the worker holds. For isolation, a skill can instead be a subagent's template
(D3).

## Consequences

- **Positive.** No new op type: the op alphabet, determinism and no-continuation invariants hold.
  HITL reuses the permission cascade's suspend and resume unchanged. The subagent boundary is
  opaque, replay-isolated and the cache strategy at once. The bench makes cost-effectiveness an
  executable assertion.
- **Negative.** An interrupt is polled between steps; a stop during a step reaches only the
  interpreters that register one (ADR-0026). The opaque `subagent_runner` re-runs wholesale on a
  parent crash; the durable forms trade that against trace visibility or a second task. Cache and
  live-cost metrics need a paid or local model run, so CI does not measure them.
- **Neutral.** The seams are keyword arguments; a bundling policy object is a later DX question.

## Open work

| item | note |
|---|---|
| better retrievers and a `k` sweep | a weak retriever behind a hard gate starved the model on HotpotQA: soft gating (warn, not block) is the candidate strategy |
| an RLM in the `decide` slot | a DSPy module or combinator policy, with the step's template tuned by `improve` (ADR-0005) |
| a cross-framework ablation | the same model, questions and scorer behind other agent frameworks, reporting accuracy and cost |
