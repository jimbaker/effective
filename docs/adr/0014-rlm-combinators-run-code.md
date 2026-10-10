# ADR-0014: RLM combinators: sandboxed code execution as an effect (`run_code`), skill scripts as pinned code, recursion as configuration

- **Date:** 2026-07-03
- **Status:** Accepted. Built: `run_code` ([`src/effective/code.py`](../../src/effective/code.py)), the Monty engine
  ([`src/effective/monty.py`](../../src/effective/monty.py)), pinned skill scripts (`Pin.script` in [`src/effective/skills.py`](../../src/effective/skills.py)),
  the `hoisted` / `recurse` / `route` / `descend` combinators ([`src/effective/combinators.py`](../../src/effective/combinators.py)),
  the structural `scoped` frame (`effective.api.scoped`), and `code_act` ([`src/effective/compose.py`](../../src/effective/compose.py)).
  Unbuilt: a second interpreter (the Pyodide fallback), a container tier, the prewarmed image.
- **Relates to:** [ADR-0002](0002-harness-layer-stack.md) (the permission cascade that actions flow through), [ADR-0008](0008-dynamic-workflows-as-ops-applicative-parallelism.md)
  (a fleet run's sealed nondeterminism, the `fork` reservation, `gather` keying), [ADR-0009](0009-durable-backend-two-regimes-taskcontext.md) (the
  cross-backend conformance pattern), [ADR-0016](0016-formalization-and-operational-semantics.md) (the scope-flattening lemma), [ADR-0022](0022-dashboard-projections-the-read-side.md) (provenance
  as a key frame).

## Context

RLM combinators (Zhang and Khattab, arXiv:2512.24601) structure trees of recursive model calls
over contexts too large, or tasks too decomposable, for one flat call. The reported upside is
large: an RLM over a frontier model scores 91.3% on BrowseComp-Plus at 1K documents where the
same model called flat scores 0.0%, with a median cost below a base call and a mean above it.
The defining affordance is that the model, or a skill, supplies **code** that slices, filters and
recombines data without paying tokens for each intermediate step.

Effective already had the recursion substrate: a nested `run_agent` is a durable, replayable
sub-generator, and `gather` is the data-parallel join with per-branch crash-resume. The missing
piece was code. The question was how to admit arbitrary code without breaking the determinism
boundary, the bookkeeper split, exactly-once actions or the no-call/cc invariant, and how to make
it compose with skills rather than bypass them.

Three facts shaped the answer.

| fact | consequence |
|---|---|
| a REPL-style RLM freezes its tool set at construction, and one `exec` is atomic-or-nothing | a world-mutating call inside it can never be exactly-once across a crash: both gaps are op-granularity gaps |
| Monty's execution model is generator-shaped: `start()` pauses at each host-function call and `resume(value)` continues | with no ambient authority, sandboxed code's only effects are host calls, so it satisfies the determinism boundary by construction |
| a skill's `content_hash` is a tree hash over body, references, scripts and assets | the pin identity already covers executable code; skill scripts need no new identity machinery |

## Decision

### 1. `run_code` is a combinator over existing ops; code is data

```python
out = yield from run_code(
    "summarize",
    code,                          # str: model-written this turn, or a pinned skill Script
    schema=Summary,                # the final expression's value, validated at the boundary
    inputs={"readings": readings}, # whole JSON values into the sandbox
    functions=("llm_query",),      # observational host functions, by name
    actions={"set_thermostat": Ack},  # world-mutating host functions, name -> result schema
)
```

No new `WorkflowOp` and no handler change: each **segment** is a checkpointed `Step` carrying the
reserved `code-execute` `CallTool` (`EXECUTE_TOOL`), which the deployment's domain interpreter
answers with a code engine (`MontyEngine` and `execute_tool` in `effective.monty`, deployment
infrastructure like the model caller). Function names appear at the workflow level; their bodies
are engine configuration. `inputs` carries whole values, checked to be JSON at entry, because a
`set` input would leak the host's hash seed into the sandbox on every re-execution. A prompt that
asks a model to write the code is an ordinary channel template; `run_code` is where the code lands.

### 2. The uniform replay rule

**Every host-call result, function and action alike, is re-bound from the record on replay; the
deterministic code between calls re-executes.** A model query is as nondeterministic as a
world-mutating call, so purity does not decide the split. The sandbox contributes determinism by
construction: Monty has no `time` or `random` module, and `datetime.now()`, `os.environ` and
`open()` surface as host calls the engine resolves, so the clock is an effect. Every value
crossing the boundary, live or re-bound, crosses in `canonical()` form (the checkpoint JSON form,
with sets sorted through `canonical_form`), so first run and re-execution see identical values.
This is replay-by-re-execution, the Temporal and Absurd model applied inside the op, and it is the
only durability mechanism (§4).

### 3. Granularity: `functions` seal, `actions` surface

| kind | how it runs | how it is recorded |
|---|---|---|
| `functions` (observational: model queries, fetches, helpers) | inside the segment | an ordered call log `{name, args digest, value}` in the segment's checkpoint, re-bound by index |
| `actions` (world-mutating) | the segment pauses; the workflow yields the call as its own `CallTool` op, then resumes the code with the result | one checkpoint per action, routed through the permission cascade |

The args digest makes a re-bind that lands on the wrong call site a loud `CodeEngineError`, never
a silent misbinding. An action is **exactly-once per committed checkpoint**: a crash after the
tool fires and before its checkpoint commits re-runs it, which is standard step semantics with
the crash window shrunk from a whole `exec` to one call. Each action asks the handler for its
`idempotency_key`, which the handler mints from the task and the action's placement, gather frames
included, so a tool whose receiver dedupes on it performs the action once across a crash
(`test_two_gather_branches_hand_an_action_tool_distinct_idempotency_keys`).

Cascade outcomes reach the code as ordinary Python: a denial re-enters the sandbox as
`PermissionError`, which the code may catch and route around; an escalation parks the task
durably between two segments on the human tier's `AwaitEvent`, with functions already run and the
world untouched. Inside a `gather` each branch parks on its own qualified approval event
(`effective.api.qualified_event_name`), so two gated branches need two approvals. On the durable
engine event names are global, so a human tier names its approval event with the run id in it.
A run requesting more than `MAX_ACTIONS` (64) actions fails as a runaway loop.

Op granularity is what makes an action expressible (recorded, gated, deniable) rather than
smuggled through a sandbox escape hatch such as Deno's `--allow-*` flags or a local interpreter,
both foreclosed.

Segment and action keys are `code:seg,{seg},{name}` and `code:action,{j},{name};tool:{tool}`. A
duplicate `name` occurrence-suffixes identically on both engines, a safety net that binds by
yield order and so is no license to reuse names.

### 4. No persisted mid-run snapshots

A mid-run `FunctionSnapshot` serialized and resumed after worker death would be a captured
continuation, rejected on the no-call/cc invariant. The snapshot is an in-worker interleaving
mechanism inside one segment, and nothing durable records it.

A **prewarmed interpreter** is a different thing: dumping Monty at a well-known init state (host
functions registered, a skill prelude evaluated) and starting every run from that image is
memoized initialization with no mid-computation control state, so it raises no invariant
tension. It is deferred as an optimization until sandbox init cost shows in the meter, which is
unlikely at Monty's microsecond cold start.

### 5. Monty first, behind an engine seam

Monty (`pydantic-monty`, pinned at 0.0.18) is the engine: generator-shaped host-call seam, no
ambient authority, microsecond startup, resource limits (`max_duration_secs`, `max_memory`,
`max_allocations`, `max_recursion_depth`) passed through `MontyEngine(limits=...)`. `effective.code`
never imports the engine, so the op contract owns the semantics and the engine owns execution; a
second engine (a Pyodide subprocess) would join behind the same reserved tool and earn a
cross-interpreter conformance suite in the shape of [ADR-0009](0009-durable-backend-two-regimes-taskcontext.md)'s cross-backend one. A
container or microVM tier for hostile code is deferred until untrusted third-party code runs:
Monty's capability model confines what code can name; it does not isolate a process. Firejail is rejected (a SUID-root
escalation surface).

Monty 0.0.18 parses a subset: `match`, `class` and generator `yield` raise a located
`NotImplementedError` at parse time, while comprehensions, the walrus and f-strings work. Skill
scripts stay inside the subset, so sandbox scripts are exempt from the repo's `match` preference
until Monty grows.

### 6. Skill scripts are pinned code; provenance is a key frame

`pin.script("summary.py")` resolves a script from the activation `Pin`: content the run recorded,
never the working tree. A pinned run executes inside `scoped(script.key)`, so the pin is a frame in
every segment's key and no provenance field rides the payload ([ADR-0022](0022-dashboard-projections-the-read-side.md)). Improvised and pinned
code are told apart by selecting on keys. The disclosed skill body documents the skill's
host-function contract, so a model writing gap code writes against named functions. Improvised
code that earns its keep is reviewed and committed to the skill's `scripts/`: agentic discovery,
then a distilled artifact, then deterministic re-application ([ADR-0008](0008-dynamic-workflows-as-ops-applicative-parallelism.md)), applied to code. The
improvised-code rate is a cost the Pareto surface watches shrink.

### 7. The RLM loop is a configuration

`run_agent(decide=..., act=code_act(...))` with recursion as a `spawn_subagent` host function is
the RLM loop, durable. The combinators are thin sugar over `gather`, `scoped`, `activate_skill`
and `run_code`:

| combinator | rule it bakes in |
|---|---|
| `recurse` | `decompose` runs before the `gather`, so the data-dependent branch count is recovered from recorded state; `combine` is a balanced tree-fold of gathers (the combine step has its own context rot) |
| `route` | uniform dispatch by a classifier, the join a frozen tool set cannot express |
| `hoisted` | skill activation is value-independent, so it runs once above the fan-out and every branch shares the recorded `Pin` |
| `descend` | a level-by-level drill that ends on the judge's answer or on budget exhaustion as a durable park (or a `Grantor`) |

`fork` stays reserved for counterfactual marginals ([ADR-0008](0008-dynamic-workflows-as-ops-applicative-parallelism.md)); the data-parallel shape here is
always `gather`.

### 8. Scope is a handler-applied frame

Namespacing is structural. `scoped(scope, body)` yields a `Scoped` op that the handler interprets
by running `body` with every key it mints prefixed by `scope`, in the same position and by the
same mechanism as a gather branch's `gather:{g},{i};` coordinate. `scope` is a `Key` composed by
the one composer. A frame ends at the term separator `;`, which `scope_prefix` refuses inside an
atom, and nesting composes by nesting: a step `s` inside `scoped(a:1, scoped(b:2, …))` records as
`a:1;b:2;step:s`. The combinators wrap each branch (`rec:{i}`, `fold:{level},{k}`, `d:{depth}`)
and `code_act` runs inside the turn's frame, so no author threads a scope through a callback or
splices one onto a name. A `run_code` name cannot forge a scope boundary no `scoped` opened,
because every key `run_code` mints leads with the `code` tag behind the `step` arm:
`scoped(rec:0, run_code("x"))` records `rec:0;step;code:seg,0,x`, while a run named `rec:0;x`
records `step;code;seg:0;rec:0;x`. `run_code` also refuses a name containing `/`, the path
delimiter no key atom may hold. The flattening's injectivity on well-formed scopes is
`flatten_injective_on` in [`formal/lean/Effective/Scopes.lean`](../../formal/lean/Effective/Scopes.lean), with the aliasing foil
`flattenBad` ([ADR-0016](0016-formalization-and-operational-semantics.md)); that model spells the separator `/`, and [ADR-0016](0016-formalization-and-operational-semantics.md) names the gap.

The rejected alternative was a scope string interpolated at the call site. It aliases: a scope
ending in the delimiter, spliced ahead of a model-controlled tool name, composes the same key as
no scope with a tool name embedding that delimiter, giving two actions one address.

## Deferred, with triggers

| piece | trigger |
|---|---|
| a second engine (Pyodide subprocess) and a cross-interpreter conformance suite | a workload Monty's subset cannot run |
| prewarmed Monty image | sandbox init cost visible in the meter |
| container or microVM tier | running untrusted third-party code |
| a measured RLM-versus-structured contrast | the evaluation track |

Mid-run snapshot persistence is rejected, revisited only if replay-by-re-execution measurably
fails a real workload, and then as an explicit invariant discussion.

## Consequences

The substrate gains RLM's explorative power as one combinator whose every effect is recorded,
gated and replayable, and skills gain an executable tier: prose that guides code, scripts that
are code, and a measured promotion path from the first to the second. The durable cases are pinned
on both engines in [`tests/test_conformance.py`](../../tests/test_conformance.py): a clean run, crash at every op with the action
exactly-once, a mid-code human park, an in-sandbox denial, disjoint keys under `gather`, and
per-branch parks under `gather`. [`tests/test_run_code.py`](../../tests/test_run_code.py) pins the cross-process replay under a
differing `PYTHONHASHSEED`. The cost is an alpha dependency held behind an engine seam, and a
standing tension, structure taxing exploration, that the Pareto surface measures.

The growth shape is O(segments × log): the cumulative function log rides every segment's args and
result. That suits glue code; a run making hundreds of large host calls wants fewer, bigger
functions.
