# Architecture map

Where each subsystem lives and what to open first. Sizes are not here: `tokei` or `wc` answers
that, and it is not what a reader arrives needing.

| subsystem | what it is | open first |
|---|---|---|
| **effective-core** | the op alphabet, authoring API, layers, channels, cost and permission, the ReAct loop, the SQLite engine and the ledger | `api.py`, `ops.py`, `channels.py`, `layers.py`, `react.py`, `sqlite.py` |
| **effective-handlers** | the three interpreters, and the single injective op-key producer they share | `handlers/{recording,replay,absurd,base}.py` |
| **effective-interpreters** | what answers a domain op: model callers, a judge, a shell, the web | `interpreters/{openai,cli,jev,shell,web}.py` |
| **effective-cards** | the view axis: a frozen `CardSpec` IR and pure renderers | `cards/spec.py`, `cards/render_*.py`, `cards/tstring.py` |
| **agent** | evaluation: Pareto scoring, bench harnesses, subagent runtimes | `runtime.py`, `eval.py`, `bench*.py` |
| **examples** | two packaged example agents, and the single-file examples | `src/examples/coder/`, `src/examples/deep_research/`, `examples/first_workflow.py` |
| **tui** | a terminal view over a durable run | `src/tui/app.py`, `just tui-demo` |
| **formal** | Lean for the pure facts, Quint for the interleaving space | `just formal`, `formal/lean/`, `formal/quint/` |
| **tests** | cross-backend conformance, and the crash/suspend/permission proofs | `_conformance.py`, `_durable.py`, `just check` |
| **infra-scripts** | the recipe index, pinned tool infra, migrations, the gate scripts | `justfile`, `infra/*/PIN.txt`, `scripts/` |

The dependency rule runs on one discriminator: which side of the effect boundary a module sits on.
What YIELDS an op is substrate and lives in `effective`; what ANSWERS one is an interpreter in
`effective.interpreters`, its vendor behind an extra and imported only by its own module.
Dependencies are one-way, from consumers to the substrate, and `just lint` checks the direction
(`effective.lint --deps`, rule `seam-dep-direction`).

### The substrate

**effective-core.** A workflow yields the closed `WorkflowOp` union (`ops.py`); handlers interpret it. Two composition seams, op-layer and domain-layer, driven by `drive_through` (`layers.py`), carry `permission` (fail-closed `cascade([rules, human])`), `cost` (`MeteredInterpreter`, the `Usage` monoid, `BudgetExceeded`) and `telemetry` (the `traced` domain layer, the OTLP span file under the GenAI conventions). The data axis is `channels.py`: the `render(t"...", output=...) -> Prompt[S]` processor with `Gated` guardrails, `Repair`, skill disclosure, and cache and role directives. `code.py`, `monty.py`, `skills.py` and `combinators.py` layer sandboxed execution and skill-pack machinery on existing ops with no new replay machinery. `run_agent` (`react.py`) is a plain-generator ReAct loop that yields `ask_llm`/`call_tool`/`await_event`, so the handlers make a whole trajectory durable and replayable: record once, replay with no model. `ledger.py` (Postgres) and `SqliteLedger` are the canonical append-only bookkeeper. `__init__.py` re-exports only the control-axis core. An await or sleep inside a `gather` branch parks on an engine whose ctx can peek an event (the embedded SQLite engine, `ConcurrentAbsurdCtx`); a ctx without that capability refuses it with `NotImplementedError`.

**effective-handlers.** One generator, three meanings. `RecordingHandler` runs with no I/O (canned responses, in-memory ledger, each op recorded to a `TraceEntry`; an unmet `AwaitEvent` parks as `Suspended`). `ReplayHandler` re-runs the generator against that trace, raising `ReplayMismatch` on divergence. `DurableHandler` (`handlers/absurd.py`) is the durable path, mapping ops onto the `TaskContext` protocol that both Absurd/Postgres and the embedded SQLite engine satisfy, with Pydantic checkpoint serde and `ConcurrentAbsurdCtx` for a gather whose tools overlap while its writes serialize. The keystone is the key pair: `op_key` (`base.py`) over `compose_key` (`effective.keys`: `grammar` holds the key itself, `marker` the `Tag`/`Segment`/`AuthorityTag` family, `processor` the composer, `frame` frames and arms, `registry` the source map) is the *single* injective key producer all three paths share, so a recorded key and a durable checkpoint key are the same string by construction. The `handlers` package facade under-states the surface: `DurableHandler`, `TaskContext` and `canonical_form` are reached by importing `effective.handlers.absurd`.

**effective-cards.** The view axis, sibling to `channels`. A card is *data*: a frozen `CardSpec` IR (`spec.py`, no intra-package imports) with a `Metric`/`Interval`/`Badge`/`Action`/`VegaChart`/`Slot` vocabulary. Pure renderers lower one spec to a self-contained HTML fragment (`render_html` plus a one-time `VEGA_BOOTSTRAP`), a Markdown fallback, a Shiny `htmltools` tag tree (`render_shiny`, the only file allowed to import htmltools), and an inspectable `manifest`. A PEP 750 face, `card(t"...")` (`tstring.py`), builds the same spec. Charts ride in as dicts, so the core imports no charting library.

**agent.** The bench and evaluation half. `runtime.py` and `compose.py` supply three subagent regimes (opaque, durable-inline, spawned-task), HITL-as-suspend, and `code_act`. `eval`, `scoring` and `effective.pareto` score a cost/quality/latency frontier; `bench` and `bench_telemetry` show that replay is free; `bracket`, `gsm8k`, `hotpotqa`, `skillsbench` and `contrastbench` are concrete harnesses.

### Examples and supporting subsystems

**examples.** `src/examples/coder` is a small coding agent: four tools, one working state, and a test suite that decides when the work is done. `src/examples/deep_research` is a frontier search over leads, stopped when two hosts settle every cell of the answer. `examples/first_workflow.py` is the workflow `docs/first-workflow.md` walks through. `examples/smol_agent.py`, `smol_door.py` and `smol_durable.py` are a house agent in miniature: the bare loop, a judged door, and the durable form with interruption.

**formal.** Machine-checked operational semantics. Lean 4 proves the pure facts durable replay rests on: key injectivity (`Keys.lean`, including a reproduced cross-gather collision), scope-string injectivity (`Scopes.lean`), step determinism and replay stability (`Step.lean`), and the append-only idempotent ledger (`Ledger.lean`). Quint model-checks the interleaving space: `gather.qnt` reproduces the suspend-in-gather totality bug and adds fairness-conditional liveness. The fast gate (`just formal`: `lake build` plus quint typecheck) takes seconds; the Apalache/TLC verification (`just formal-verify`) takes minutes and runs on demand. Open edges: gather-by-scope composition, determinism and append-only for compound ops, and an obligation on serialized writes that only tests discharge.

**tests.** The ground truth for what is built; no `src/` module imports `tests/`. The core: `_conformance.py` (one workflow set, identical assertions on SQLite and Absurd/Postgres through the same handler) and `_durable.py` (crash-at-every-op, suspend/resume, permission surviving worker death). Coverage is always reported; the 90% gate (`just cov`) runs against real Postgres. **A test carries a role** (unit, spine, journey, adversarial, conformance), and the role decides what a passing run proves: *minimize overlap* applies to the unit role only, and an adversarial pass counts only if the test could have failed.

**infra-scripts.** The operational surface at the top of the dependency graph; nothing imports it back. The `justfile` is the recipe index; `scripts/setup_absurd.sh` installs the pinned schema; the gate scripts (`link_check.py`, `wiki_lint.py`, `fstring_sweep.py`, `key_sweep.py`, `preflight.py`) are what `just check` runs. Supply-chain discipline lives in pinned `infra/<tool>/` directories: `absurd` and `tdom` vendored, `pg-test` and `redis-test` digest-pinned, `formal` a hash-gated sandbox.
