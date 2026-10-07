# Index

The catalog of the wiki and the docs. The wiki holds the arguments: each concept page says what is
true now and why, and is rewritten in place when that changes. The docs hold the maintained
deliverables written from it.

## Docs

| document | what it holds |
|---|---|
| `docs/intro.md` | the model in ten minutes: an agent loop as a generator, the handler, the tape, layers as hooks, combinators |
| `docs/first-workflow.md` | a first workflow on the package, recorded and replayed with no model |
| `docs/effective-101.md` | the core concepts in order: the two seams, the op set, the author surface, the combinator algebra, the temporal shapes, the t-string grammars, the read side, the invariants |
| `docs/effective-design.md` | the substrate written down whole: the op alphabet, the small-step semantics the handlers refine, the identity grammar, the combinator algebra, the channel processor, the formal estate |
| `docs/repertoire-axis.md` | the repertoire axis: declare the vocabulary a workflow may draw on, resolve it against a deployment, record what was resolved |
| `docs/README.md` | what `docs/` and `wiki/` each hold, and what to open first |

Read `docs/effective-design.md` when you need the semantics rather than a subsystem, or before
proposing anything that changes what a handler owes an op.

## Decisions

The architecture decision records in `docs/adr/`, each the definitive statement of one decision.
A document cites one by number (`ADR-0020 §3`), so a citation survives a rename; code, tests and
models cite none, and state their invariants in their own words. Numbers with no row have no record.

| ADR | status | decision |
|---|---|---|
| 0001 | Accepted | the channel processor: a typed function on `Template` |
| 0002 | Accepted | the harness as a layer stack: two seams, one combinator idiom |
| 0004 | Accepted | the ReAct loop as a durable driver over a typed step |
| 0005 | Accepted | the optimizer is a coding agent in a loop |
| 0006 | Proposed | two-stage marginal selection, promoted by Pareto dominance |
| 0007 | Proposed | channel discovery by filesystem convention and a typed manifest |
| 0008 | Accepted | dynamic workflows as agent-layer ops; applicative parallelism (`gather`) |
| 0009 | Accepted | the durable backend: two regimes over one `TaskContext` |
| 0010 | Accepted | the card `CardSpec` IR, rendered to many targets |
| 0011 | Accepted | a Shiny-native render target for the card IR |
| 0014 | Accepted | RLM combinators: sandboxed code execution as an effect |
| 0016 | Accepted | formalization: three proof tiers and the operational semantics |
| 0017 | Accepted | tunable seams take data; closures are for fixed policy |
| 0018 | Accepted | measured-spend accrual |
| 0019 | Accepted | `govern` and `serve`, the two composition combinators |
| 0020 | Accepted | one grammar for identity: structurally-aware key composition |
| 0021 | Accepted | dynamic graphs: three growth axes, one projection |
| 0022 | Accepted | dashboard projections: the read side of the graph |
| 0023 | Accepted | test roles: what a test is for decides what its pass proves |
| 0024 | Proposed | the walk: one mint, many interpreters |
| 0025 | Accepted | race and quorum |
| 0026 | Accepted | in-flight cancellation |

## Concepts

| page | what it holds |
|---|---|
| [concepts/architecture](concepts/architecture.md) | where each subsystem lives, what to open first, and the dependency direction |
| [concepts/machine](concepts/machine.md) | the EFSM shape, its two refusals, and why durable suspend is replay |
| [concepts/tapes](concepts/tapes.md) | three tapes joined by one address, two disciplines, and where the order obligations land |
| [concepts/graph](concepts/graph.md) | three projections that are two relational operations, over a run tape |
| [concepts/flatten](concepts/flatten.md) | why a `Template` has no `__str__`, where each grammar's processor lives, and the one position an f-string is still right |
| [concepts/python-314](concepts/python-314.md) | read 3.14 syntax as a feature; verify before "fixing" it |
| [concepts/judgment](concepts/judgment.md) | the `Judge` op and its interpreters: the template as the request, what the probes measured, retrieval and replay |
| [concepts/sources](concepts/sources.md) | a fact carries the source that asserted it: the tiers, a source as a tool, authority by corroboration, reuse by content |
| [concepts/recursion-shapes](concepts/recursion-shapes.md) | classic shapes as semantic probes, classified by the topology they add and the error kinds they need |
| [concepts/building-with-effective](concepts/building-with-effective.md) | what to build: real applications mapped onto the shapes, runnable startup workflows, and the boundaries |
| [concepts/watchers](concepts/watchers.md) | supervisor, watchdog, monitor and deadline, split by what each one watches |
| [concepts/bounded-wait](concepts/bounded-wait.md) | what decides a wait that names a deadline: the ordered arms, the wake's lifetime, the slot, and where it is refused |
| [concepts/forced-schedules](concepts/forced-schedules.md) | what a schedule instrument can decide and what it cannot, and why a race is outside it |
| [concepts/witness-quotient](concepts/witness-quotient.md) | the admission witness as a quotient of what a checkpoint keeps, its one runtime obligation, and the classes it merges |
| [concepts/enforcer-domain](concepts/enforcer-domain.md) | a gate is bounded by what it scans, with worked cases and what a zero really means |
| [concepts/testing](concepts/testing.md) | testing a workflow: an instrument per question, replay as the strict judge, why production resume is lenient, the test roles |
| [concepts/evidence](concepts/evidence.md) | a ratification is a claim, an instrument is a claim, and a green test can be weaker than it looks |

## References

[references](references.md): work this project builds on, follows or ships, with full citations, and background
reading. Source code names that work by a bare proper noun; the page resolves it.
