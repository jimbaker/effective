# ADR-0002: The harness as a layer stack: two seams, one combinator idiom

- **Date:** 2026-06-13
- **Status:** Accepted. Built in [`src/effective/layers.py`](../../src/effective/layers.py) (`drive_through`, `op_layer`,
  `domain_layer`, `OpLayer`, `DomainLayer`, `compose_ops`, `compose_domain`, `check_seam`,
  `retry`, `retry_domain`), the layer-authority rules in `effective.lint --layers`, `metered` and
  `traced` as domain layers, and op-layer stacks on `RecordingHandler` and `DurableHandler`.
  Permission and the merged gate are built on the op seam (`cascade`, [ADR-0019](0019-govern-serve-composition-combinators.md)'s `govern`). Not
  built: memory hydration or recall, and strategy-as-handler.

## Context

The agent-harness literature (Akshay Pachaar, *The Anatomy of an Agent Harness*) lists about
twelve components of a production harness: orchestration loop, tools, memory, context
management, prompt construction, output parsing, state management, error handling, guardrails,
verification loops, subagent orchestration, lifecycle. The list is flat and gives no rule for
where each component lives.

Effective already reifies effects on three axes, and each has a named seam:

| axis | what it carries | where |
|---|---|---|
| control | the ops a workflow yields (`WorkflowOp`), driven by a handler's drive loop | [`src/effective/ops.py`](../../src/effective/ops.py) |
| interpreter | cross-cutting layers over the op stream: cost, cache, telemetry, retry, permission | [`src/effective/layers.py`](../../src/effective/layers.py) |
| data | the typed I/O of one model call: prompt construction, output parsing, guardrails | [`src/effective/channels.py`](../../src/effective/channels.py) ([ADR-0001](0001-channel-processor.md)) |

So the twelve factors are three kinds. A harness is `(op set) x (layer stack) x (channel set)`,
sequenced by a loop, and the loop is the composition operator rather than a member of the stack
([ADR-0004](0004-react-loop-driver.md)).

### The forcing function

The determinism boundary decides which seam a factor belongs to:

> A factor that introduces nondeterminism or alters control flow **must** be reified, as a
> yielded op or as a *recorded* handler decision. A factor that only observes or rewrites model
> and tool I/O **may** be a silent layer. A factor that shapes one call's typed schema is a
> channel.

Retry alters control flow; telemetry observes; a guardrail shapes one call's schema. Compaction
rewrites the transcript, and the transcript is workflow state between yields, where no layer
reaches, so `run_agent` records compaction as its own `react:compact` step ([ADR-0004](0004-react-loop-driver.md)).

### Two seams, deliberately not unified

| | domain seam | op seam |
|---|---|---|
| wraps | the domain interpreter a `Step` dispatches to | the handler's drive loop |
| alphabet | `DomainOp`: `AskLLM`, `Judge`, `CallTool` | the full `WorkflowOp` union, `AwaitEvent` included |
| lifetime | call-scoped | stream-scoped |
| authority | cannot suspend or alter control flow | may park for a human, inject ops, sleep |

The alternative is to unify the alphabets: promote `AskLLM` and `CallTool` to `WorkflowOp`, one
`perform`, one onion. It loses on three concrete grounds:

1. **Authority.** A metering layer that *could* park the workflow is a hazard. The domain seam is
   confined so it cannot alter control flow.
2. **Replay grain.** `op_key` keys the trace and the checkpoints on `WorkflowOp`s. Keeping the
   model call a `DomainOp` inside `Step` keeps the step the durable grain; unifying floods the
   key space.
3. **Lifetime.** Call-scoped and stream-scoped layers have different bases.

The design is two effect alphabets and one bridging idiom, the way `contextlib` adapts a
generator to the context-manager protocol without merging the two concepts.

## Decision

Build a composable layer stack at both seams, keeping the seams distinct in type, in lint and at
run time.

| item | decision |
|---|---|
| L1 layer shape | a layer is generator middleware: `result = yield op` forwards the (possibly rewritten) op inward and receives the result. Setup before, teardown after, in reading order, as in `@contextmanager` |
| L2 multi-yield | a layer may yield more than once, which `@contextmanager` forbids; a loop around `yield op` is retry |
| L3 trampoline | `drive_through(layers, op, base)` pumps the stack: `.send` threads results up, `.throw` delivers a downstream exception into the layer's `yield`, and a layer that returns before yielding short-circuits |
| L4 two decorators and two types | `@op_layer`/`OpLayer[T]` and `@domain_layer`/`DomainLayer[T]`, distinct by alphabet, authority and lifetime |
| L5 two combinators | `compose_domain(layers, base)` returns an interpreter; `compose_ops(layers, handler)` sets the handler's `op_layers`. Selecting a harness is editing two lists |
| L6 lint | the authority model is a lint rule keyed on the decorator |
| L7 first domain layer | the cost concern as `metered`, a `@domain_layer` |
| L8 first op layer | `retry`, the multi-yield proof |

A domain layer, from [`src/effective/layers.py`](../../src/effective/layers.py):

```python
@domain_layer
def run(op: DomainOp[Any]) -> Generator[DomainOp[Any], Any, Any]:
    for attempt in range(attempts + 1):
        try:
            return (yield op)
        except on as exc:
            if (delay := wait_before_retry(exc, attempt, attempts, backoff)) is None:
                raise
            if delay:
                sleep(delay)
```

### Composing a harness

```python
domain = compose_domain([metered(accrue), traced(sink)], base)
handler = compose_ops([cascade(tiers)], RecordingHandler(responses))
```

Adding a factor changes one of the two lists and never `run_agent` or the workflow. When a better
model does not need a layer, removing it is a one-line deletion.

### Which seam: three questions in order

| question | yes means |
|---|---|
| must it see the `WorkflowOp` alphabet (step names, `AwaitEvent`)? | op seam |
| may it park, suspending for a grant or an approval? | op seam (`govern`, [ADR-0019](0019-govern-serve-composition-combinators.md)) |
| may it block for long (a sleep that should release the worker)? | op seam, as a `SleepUntil` |

Otherwise it transforms, re-invokes or observes one call, and it is a domain-seam service
(`serve`, [ADR-0019](0019-govern-serve-composition-combinators.md)). Raising `Refused` is not parking; a domain layer may raise.

### Lint and assembly guards

| guard | what it refuses |
|---|---|
| `domain-layer-no-control-flow` | a `@domain_layer` yielding a `WorkflowOp` constructor |
| `op-layer-alphabet` | an `@op_layer` yielding a bare `DomainOp` constructor, which would bypass `Step` and the durable grain |
| `op-layer-no-broad-except` | an `@op_layer` catching `Exception`, `BaseException` or bare `except`, which would swallow the durable suspend signal |
| `check_seam` | a layer marked for one seam installed at the other, at assembly time. Such a layer runs on a live pass and observes nothing on replay, so it is refused before it runs |

### Which ops reach the op-layer stack

`layer_routing` partitions `WorkflowOp` totally:

| routing | ops |
|---|---|
| `layered` | `Step`, `SleepUntil`, `AppendLedgerRow`, `StoreArtifact` |
| `layered-durable-only` | `AwaitEvent`: the recording core decides park or resume before the stack, because parking mid-layer would need a captured continuation |
| `unlayered` | `Gather`, `Race`, `Scoped`, `Respawn`: their layers apply within each branch or body |

A layer-injected park (a permission `AwaitEvent`) resolves on replay: the durable handler re-runs
the layer with the recorded answer folded in, and never resumes a captured layer stack.

### Retry at each seam

`retry` (op seam) is proven on a context that keeps no durable checkpoints
(`test_retry_op_layer_recovers_a_flaky_step_through_the_handler`). On the durable engines it is unsound
under retry-then-crash: a re-forward calls `ctx.step(name)` again, which advances the engine's
occurrence counter, so the retried success commits under `name#2` and a crash-replay that looks
up `name` re-executes the call. Retry of domain I/O therefore uses `retry_domain`, which
re-invokes the call inside one `ctx.step` and is invisible to checkpoints, the trace and the
ledger. A `RateLimited` (429) is retried only when a `backoff` is configured, so an immediate
re-fire on a rate limit cannot be spelled.

## Consequences

**Positive**

- Each harness factor sorts into op, op layer, domain layer or channel by the determinism
  boundary.
- Harness thickness is a dial; thinning is a deletion from a list.
- The authority model is checked: a domain layer cannot suspend the agent, by lint rule.
- Cost, telemetry, cache and retry are instances of one shape rather than bespoke classes;
  `MeteredInterpreter` is `metered` over a base, assembled by `compose_domain`.

**Risks and their mitigations**

| risk | mitigation |
|---|---|
| the trampoline carries the `.send`/`.throw` complexity `@contextmanager` hides | written once and unit-tested in isolation ([`tests/test_layers.py`](../../tests/test_layers.py): rewrite, multi-yield, throw-into-yield, early return, empty stack); authors see a linear generator |
| op-seam retry and occurrence-keyed checkpoints | `retry_domain` below the seam for domain I/O |
| two decorators are more surface than one | the surface buys the lint handle, the type meaning and authority confinement |
| layer order is semantic: retry outside a budget differs from a budget outside retry | order is explicit in the list, outermost first, and tested (`test_layer_order_is_outer_to_inner`) |

## Invariants

- **Determinism boundary.** Layers are pure between yields and the trampoline performs no I/O.
- **No first-class continuations.** A park from a layer resolves by replay.
- **DB-free core.** `effective.layers` imports no SQLModel.

## Alternatives rejected

- **One unified op-level middleware.** Forfeits capability confinement, changes the replay grain
  and conflates two lifetimes.
- **One shared `@layer` decorator.** Forfeits the lint discriminator, the type meaning and
  authority confinement.
- **Cross-cutting concerns as bespoke classes.** Works for one concern; does not compose, and
  gives the op-seam factors no place to live.
- **Strategy as a layer (ReAct against plan-and-execute).** The loop is the composition operator;
  strategy stays a program ([ADR-0004](0004-react-loop-driver.md)).

## Open questions

- **Memory.** Is recall a yielded op (in the trajectory, replayable) or a domain layer that
  hydrates the prompt (invisible)? Treating memory as a hint to verify against state before
  acting pushes recall toward an op.
- **Declared ordering.** Should `compose_ops` and `compose_domain` validate a declared order
  (retry must wrap budget), or is list order the whole contract?
