# ADR-0017: Tunable seams take data; closures are for fixed policy

- **Date:** 2026-07-07
- **Status:** Accepted, as a convention held at review; no lint enforces it. Built: the first data
  twin, `allow_table(PermitPolicy)` in `src/effective/permission.py`, tuned by `improve` in
  `src/agent/permit_tuning.py` (`tests/test_permit_tuning.py`). The closure-shaped seams listed
  under "Conversion backlog" remain closures.
- **Relates to:** ADR-0005 (the optimizer that needs data seams), ADR-0001 (`Gated.predicate` is
  the channel-axis instance of the gap), ADR-0002 (the permission cascade, where `allow_table`
  lands).

## Context

ADR-0005 makes `improve` the default optimizer, and an optimizer can only mutate data. `improve`
candidates flow through recorded `propose` and `score` ops, so a candidate must be a value the
optimizer can hold in its search pool, re-derive on replay from the recorded `propose` ops, and
diff to describe a mutation. A closure is none of these: it neither serializes, diffs nor mutates.

The substrate has a systematic and otherwise good idiom of expressing a seam's policy as a closure
passed to a factory:

| seam | closure |
|---|---|
| `rules(policy)` in `permission.py` | `policy: Callable[[WorkflowOp], Verdict]`, a permission tier from a pure function |
| `Gated(schema, predicate, reason)` in `channels.py` | `predicate: Callable[[T], bool]`, a guardrail from a per-site predicate |
| `FormGate(predicate, reason)` in `channels.py` | `predicate: Callable[[Mapping[str, Any]], bool]`, a cross-field constraint |
| `improve(..., done=..., compact=...)` in `improve.py` | the loop's own termination and compaction triggers |

Each composes at run time and is invisible to the optimizer: `improve` cannot checkpoint a closure,
diff two of them to describe a mutation, or render one as feedback for a proposer to reflect on.
These seams are swappable by a person and not optimizable by the machine, which caps the ADR-0005
thesis (every reasonable seam is an optimization target) at the callable boundary.

The repo had already found the fix once. `TypedField` in `channels.py` carries "the schema,
constraints and any canonicalisation" on an `Annotated` type: a guard expressed as data rather than
as a per-site lambda. This ADR generalizes that move.

## Decision

**A seam intended to be tunable takes a data form interpreted by a fixed function; a closure is
reserved for fixed policy the optimizer must not touch.** The data form is at once the optimizer's
mutation surface (ADR-0005) and the value that checkpoints for replay: the two requirements are the
same requirement.

A tunable seam satisfies a reachability checklist derived from `improve`'s contract:

| requirement | meaning |
|---|---|
| config as value | the tunable is data (a Pydantic model or a primitive) the optimizer can hold, diff and re-derive on replay from its recorded `propose` ops |
| scorable objective | a `Score[C]`, `config -> Effect[Measurement]`, exists, ideally offline over recorded history |
| recorded propose | a mutation is itself a recorded op (an `ask_llm` over the data representation), so the search is replay-exact |

The rule of thumb: `allow_table(PermitPolicy)` rather than `rules(lambda op: ...)`; a constraint
parameter on an `Annotated` type rather than an opaque `Gated` predicate; a threshold field rather
than a `done` or `compact` callback, wherever the policy is meant to be tuned.

## The first data twin

`PermitPolicy` is a Pydantic model, an allow-set of op-key prefixes, and `allow_table(policy)` is
its fixed interpreter. It checkpoints, diffs, and is mutated by `improve`. `agent.permit_tuning`
tunes it by offline replay of a recorded `(op key, human verdict)` corpus, with `false_allow`
minimized as the hard safety objective and `auto_rate` maximized, at zero live risk and zero model
calls. `allow_table` answers Allow or Escalate and never Deny, so a tuned allow-set can only widen
what auto-settles and never block.

## Conversion backlog

Each is a closure today and gets a data twin when a tuning need is real.

| seam | data twin |
|---|---|
| `rules(policy)` | a data policy for the tunable allow and deny rules; fixed safety rules stay closures |
| `Gated.predicate`, `FormGate.predicate` | a declarative constraint on the `Annotated` type (the `TypedField` shape); the prompt text around the channel is already data and already reachable |
| `improve`'s `done` and `compact` | threshold fields where the trigger is a scalar over recorded state (a token budget, a round count), keeping the callback for bespoke logic |

## Consequences

More seams become `improve`-reachable, and the ADR-0005 thesis extends past the callable boundary.
The labeled corpus accumulates as a byproduct: every verdict, repair and denial is a recorded op,
so the tuning data for each data seam builds up in normal operation (every human approval decision
is already a `PermitPolicy` label). Data twins can be inspected and compared as diffs, so a caller promotes a
config digest to the ledger rather than an opaque function.

| limit | mitigation |
|---|---|
| not type-enforceable: a `Callable` parameter cannot know its caller wanted it optimizable | a convention the seam author opts into, held at review |
| verbosity: a data model and a fixed interpreter is more code than a lambda | apply it only to seams intended to be tuned |
| premature twins: converting a seam nobody tunes is speculative | convert on a demonstrated tuning need, as `allow_table` was |

## Alternatives rejected

| alternative | why it lost |
|---|---|
| serialize the closure (pickle or marshal a lambda) | a serialized closure still does not diff or mutate, and cross-process pickling of code is a supply-chain and determinism hazard |
| force it in the type system | not expressible: a `Callable`'s optimizability is the caller's intent, invisible to the callee's signature |
| leave the seams as closures and declare that `improve` reaches data seams only | concedes the compositionality thesis and leaves the highest-value seams (permission policy, channel guards) un-tunable |

## Invariants

- **Tunable implies data.** A seam documented as `improve`-tunable exposes its policy as
  checkpointable data interpreted by a fixed function.
- **Fixed safety stays fixed.** A conversion never turns a safety rule into a tunable one:
  `allow_table` widens auto-settling and cannot deny; deny rules and the human tier stay where the
  optimizer cannot touch them.
- **Conversion follows need.** A data twin is added on a demonstrated tuning need.
