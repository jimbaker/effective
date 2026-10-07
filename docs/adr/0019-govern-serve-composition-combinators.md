# ADR-0019: Two composition combinators: `govern` (control gate) and `serve` (call transforms)

- **Date:** 2026-07-19
- **Status:** Accepted. Both are built: `govern` in [`src/effective/govern.py`](../../src/effective/govern.py), `serve` and its
  `Service` alias in [`src/effective/cost.py`](../../src/effective/cost.py), with `retry_domain` ([`src/effective/layers.py`](../../src/effective/layers.py)) as
  `serve`'s first service. `govern` is formalized in Lean and Quint (§7).
- **Extends:** [ADR-0002](0002-harness-layer-stack.md) (the harness-layer stack: the two seams and `drive_through`).

## Context

Budget, meter, permission and retry are all cross-cutting factors interpreted over the same op
stream. The question is whether they share one interpreter loop.

[ADR-0002](0002-harness-layer-stack.md) gives **one loop mechanism**, `drive_through`, the trampoline that pumps a per-op layer
stack down to a base, and **one writing idiom**, a generator middleware `result = yield op`. A
layer may yield more than once, so a `for` or `while` around `yield op` is retry. It also gives
**two seams**, kept apart on purpose:

| seam | decorator | alphabet | lifetime | authority |
|---|---|---|---|---|
| op | `@op_layer` | `WorkflowOp` | stream-scoped | may suspend and inject control-flow ops |
| domain | `@domain_layer` | `DomainOp` | call-scoped | cannot alter control flow |

Two facts make the split structural. Permission and the budget trip are one pattern: a
`Proceed | Park | Refuse` transition with a durable await. And retry on the op seam is unsound on
the durable engines: Absurd's `begin_step` advances the per-run occurrence counter on every
attempt, so a retry followed by a crash commits under `name#2` and replay re-executes the call
live. Retry belongs below the checkpoint, on the domain seam.

## Decision

### 1. One trampoline, one idiom, two seams

The two seams stand, and the [`src/effective/layers.py`](../../src/effective/layers.py) header carries the rule for placing a new
factor as three questions asked in order:

| question | yes means |
|---|---|
| must it see the `WorkflowOp` alphabet (a `Step` or `AwaitEvent` name)? | op seam |
| may it park, suspending for a grant or an approval? | op seam (`govern`) |
| may it block a durable worker for long? | op seam, as a `SleepUntil` |

Otherwise it transforms, re-invokes or observes one call, and it is a domain-seam service
(`serve`). Refusing by raising is available at both seams; parking is the op seam's alone.

### 2. `govern(*policies)`: the control gate, one park, no cascade

A `govern` gate is an op-seam layer that asks every **policy** the same question about one op and
combines the answers into one `Verdict = Proceed | Park | Refuse`.

```python
@runtime_checkable
class Policy(Protocol):
    def __call__(self, op: WorkflowOp, state: GateState) -> Verdict: ...

def govern(*policies: Policy, gate: str, run_id: str, schema: type = Resolution,
           announce: Callable[[Park, GateState], WorkflowOp] | None = None,
           max_passes: int = DEFAULT_MAX_PASSES) -> OpLayer[Any]: ...
```

Per op, the gate folds the verdicts with `combine` and realizes the ruling:

| ruling | the gate |
|---|---|
| `Proceed` | forwards the op |
| `Refuse` | raises `Refused` (or `BudgetRefused`, below) |
| `Park` | yields `announce(park, state)` if given, then one `AwaitEvent` on `state.park_name`, absorbs the `Resolution`, and asks again |

- **Policies are peers.** `combine` is conjunctive: any `Refuse` refuses, else any `Park` parks,
  else `Proceed`. The ruling is order-free; the fused asks keep argument order so a merged prompt
  reads in assembly order. Tiered escalation inside one policy, such as permission's rules then
  human, is that policy's business (`permission.as_policy`), and `govern` adds no escalation axis.
- **One park, merged.** Several parking policies fuse their `Ask`s into one `Park`, so the gate
  yields one `AwaitEvent` and one `Resolution` carries every answer, keyed by policy name. A human
  answers once for one decision, and value-of-information prices the whole gate's counterfactual
  at one point.
- **The park name is op-scoped.** It carries the op's placed key and its per-run occurrence, so one
  settlement cannot authorize a later op.
- **Permission and budget are both policies.** `permission.as_policy` and `budget.as_policy` put
  them on one transition, which one Quint model covers (§7).

### 3. `serve(*services)`: the call transforms, an ordered queue, never a park

```python
type Service = DomainLayer[Any]

def serve(*services: Service, base: MeteredDomain) -> MeteredDomain: ...
```

`serve` composes domain-seam services around one metered call through `drive_through`, and
preserves `run_metered`, which a `.run`-only `compose_domain` wrapper would hide (disabling the
budget trip). The leftmost service is outermost. The services thread the `(result, usage)` pair:

| service | authority mode | in the tree |
|---|---|---|
| retry | re-invoke the call on a `TransientError` | `retry_domain(attempts, on, backoff=…)` |
| observe | emit a span, return the result unchanged | `telemetry.traced(sink)` |
| short-circuit | answer from a store before asking the base | `cache.Cache(...).over(base)`, a `MeteredDomain` wrapper |
| fold | accrue usage on the return path | `cost.metered`, a `compose_domain` layer that unwraps to the bare result, so it does not thread the pair `serve` needs |

`retry_domain` re-fires a rate limit (`RateLimited`) only when given a `backoff`, so "retry a 429
immediately" cannot be spelled; a wait long enough to matter belongs on the op seam as a
`SleepUntil`, which releases the worker.

### 4. The polarity is the decision rule and the composition law

| | `govern` | `serve` |
|---|---|---|
| stance | authority over the op | service to the op |
| can it park? | yes (`Park`) | no: it may raise, short-circuit or observe |
| output | a `Verdict` | the `(result, usage)` pair, or a raise |
| composition | a council of peers, order-free (`combine`) | a queue, first come first served |
| element | a policy, which decides | a service, which does |
| arity 0 | refused at assembly: a gate with no policies permits everything | returns the base unchanged |

`serve`'s argument order is its semantics: `serve(retry, cache, meter)` would retry the cache and
the call together, and a cache hit would skip the meter, so hits are free; put the meter outside the cache
to bill saved cost. Naming it *serving* makes the order a conscious choice.

### 5. Callables, typed for the reader

`Policy` is a `runtime_checkable` `Protocol` because it has a `Verdict` return to constrain.
`Service` is a type alias for `DomainLayer[Any]`: a `Protocol` with no constraint matched any
callable and typed nothing. Extension is "write a callable of the shape", with no registry and no
inheritance. State lives in a closure (`retry_domain`'s `attempts`) or a bound object
(`Cache.store`); a bound object is the better seam for a factor you will tune or A/B ([ADR-0017](0017-tunable-seams-take-data.md)),
since it exposes its state.

### 6. Budget is a composite

Budget is a meter on the domain seam ([`src/effective/cost.py`](../../src/effective/cost.py), folding usage) feeding a budget
policy on the op seam ([`src/effective/budget.py`](../../src/effective/budget.py), tripping and parking on the accrued value). A
governed resource is a domain meter plus an op policy that reads it, so budget lives in two files
by decomposition.

### 7. The refusal family, and the formal checks

A verdict names a state, an exception names the raise, and each refusing verdict maps to exactly
one exception:

| verdict | realized by |
|---|---|
| `govern.Refuse` | raising `govern.Refused`; `govern.BudgetRefused` (a `Refused`) when it carries an `Exceeded` |
| `budget.Cleared` / `Parked` / `govern.Exceeded` | the measured trip's outcomes, realized in-process by `measured_drive` and durably by the handler |
| none | `cost.BudgetExceeded`, raised by the `metered` domain layer when its `CostBudget` is spent |

`Refused` lives in `effective.govern`, the gate's vocabulary, and `permission` re-exports it. A
`Refuse` is a decision that could have parked; a `serve` raise has no recourse (retry exhausted, a
hard cap). A factor that offers recourse is a policy.

| check | what it holds |
|---|---|
| [`formal/lean/Effective/Govern.lean`](../../formal/lean/Effective/Govern.lean) | `combine` as one total function: refuse dominates park, no silent proceed, asks fuse in argument order, an empty refusal still refuses, and `combine_kind_perm_invariant` (the ruling is permutation-invariant, the polarity row of §4) |
| [`formal/lean/Effective/Decide.lean`](../../formal/lean/Effective/Decide.lean) | permission's cascade: fail-closed, and `suffix_absorbed`, which licenses the driver's short-circuit |
| [`formal/lean/Effective/EnforceMeasured.lean`](../../formal/lean/Effective/EnforceMeasured.lean) | the run-level overshoot bound `spend ≤ limit + Σgrants + c_max`, with a tightness witness |
| [`formal/quint/govern_park.qnt`](../../formal/quint/govern_park.qnt) | park, deliver, resume and worker death, one model covering both gates |
| [`tests/test_govern_conformance.py`](../../tests/test_govern_conformance.py) | the Python `combine` over the Lean model's rows |
| [`tests/test_govern_durable.py`](../../tests/test_govern_durable.py) | the merged park on Absurd and Postgres, including worker death |

`max_passes` bounds a gate that never settles: it refuses after that many resolutions instead of
parking forever, which also gives the Quint model a finite reachability question.

## The data axis: `TypeForm`

The control axis (ops, `govern`, `serve`) is a `Protocol` story; the data axis (t-strings,
channels) is a `TypeForm` story. PEP 747 (`typing.TypeForm`, Python 3.15) generalizes `type[T]` to
any type expression, so a channel's schema may be `list[Item]` or `Item | Error`, which `type[T]`
cannot spell. `src/` targets 3.14 and keeps `type[T]` until 3.15 is the pinned interpreter.

## Consequences

- Workflows do not change. Authors write `yield from ask_llm(...)`; `govern` and `serve` are
  handler assembly.
- A `serve` stack is metered-only: its `.run` raises a named error, and `DurableHandler` rejects
  one at assembly where it would be asked to run a `CallTool` or a default-contract model call.
- `serve` rejects an `@op_layer` service at composition, naming `retry_domain`.
- `serve` shape-guards the `(result, usage)` pipe: a service that returns a bare value raises a
  `TypeError` instead of tuple-unpacking a string.

## Alternatives considered

| alternative | why it lost |
|---|---|
| one unified transition for all four factors | retry is a re-entry and meter a fold on the return path; a verdict-transition reaching into the domain call is the authority creep the durable engine punishes, and it never sees `(result, usage)` |
| op-seam retry threaded into `measured_drive` | unsound on the durable engines (§Context) |
| a plain `compose_domain` retry wrapper | `compose_domain` exposes only `.run`, so wrapping a metered base hides `run_metered` and silently disables the cap |
| one gate per policy, parking in sequence | two parks price two unrelated questions and make a human answer twice for one decision |
| the name `attend` (or `mediate`, `intercept`) | `attend` fits pure observers better; `serve` was chosen because "first come, first served" teaches the ordering law and `govern`/`serve` is a legible polarity |
