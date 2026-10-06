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
| [[concepts/architecture]] | where each subsystem lives, what to open first, and the dependency direction |
| [[concepts/machine]] | the EFSM shape, its two refusals, and why durable suspend is replay |
| [[concepts/tapes]] | three tapes joined by one address, two disciplines, and where the order obligations land |
| [[concepts/graph]] | three projections that are two relational operations, over a run tape |
| [[concepts/flatten]] | why a `Template` has no `__str__`, where each grammar's processor lives, and the one position an f-string is still right |
| [[concepts/python-314]] | read 3.14 syntax as a feature; verify before "fixing" it |
| [[concepts/judgment]] | the `Judge` op and its interpreters: the template as the request, what the probes measured, retrieval and replay |
| [[concepts/sources]] | a fact carries the source that asserted it: the tiers, a source as a tool, authority by corroboration, reuse by content |
| [[concepts/recursion-shapes]] | classic shapes as semantic probes, classified by the topology they add and the error kinds they need |
| [[concepts/watchers]] | supervisor, watchdog, monitor and deadline, split by what each one watches |
| [[concepts/bounded-wait]] | what decides a wait that names a deadline: the ordered arms, the wake's lifetime, the slot, and where it is refused |
| [[concepts/forced-schedules]] | what a schedule instrument can decide and what it cannot, and why a race is outside it |
| [[concepts/witness-quotient]] | the admission witness as a quotient of what a checkpoint keeps, its one runtime obligation, and the classes it merges |
| [[concepts/enforcer-domain]] | a gate is bounded by what it scans, with worked cases and what a zero really means |
| [[concepts/testing]] | testing a workflow: an instrument per question, replay as the strict judge, why production resume is lenient, the test roles |
| [[concepts/evidence]] | a ratification is a claim, an instrument is a claim, and a green test can be weaker than it looks |

## References

[[references]]: work this project builds on, follows or ships, with full citations, and background
reading. Source code names that work by a bare proper noun; the page resolves it.
