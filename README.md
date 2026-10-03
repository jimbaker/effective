# Effective

Durable workflows and agents on algebraic effects, with PEP 750 t-strings as typed I/O channels.

A workflow is a plain Python generator that **yields typed op descriptions**: ask a model, call a
tool, wait for a human, append to the ledger. It performs none of them. A **handler** decides what
each op means, so the same generator is recorded in a test, replayed with no model, or run durably,
where a crashed worker resumes by replaying the steps it already took.

The prompts are **t-strings**. A prompt is a `t"..."` whose interpolations are typed channels:
inputs render in, and outputs declare the schema the answer must parse into. A `Gated` output runs a
check over the answer, which is a guardrail on the data boundary.

```python
def set_temperature(request_id: str, request: str, ceiling: float = 30) -> Effect[float | Repair]:
    celsius = Gated(float, lambda c: 10 <= c <= ceiling, "celsius outside the allowed range")
    prompt = t"The occupant said: {request}\nSet the thermostat to {celsius}"
    setpoint = render(prompt, output=Setpoint)
    answer = yield from ask_llm("setpoint", setpoint.messages, dict)
    match setpoint.resolve(answer):
        case Repair() as repair:
            return repair
        case Setpoint(celsius=target):
            yield from call_tool("thermostat", {"celsius": target}, str)
            ...
```

That is `examples/first_workflow.py`, which `docs/first-workflow.md` walks through line by line.

## Quickstart

Clone with git (the docs gate reads `git ls-files`, so a ZIP download fails it), then:

```bash
uv sync                                    # Python 3.14 and the development environment
uv run python examples/first_workflow.py   # record, replay, a guardrail, and replay refusing drift
```

`first_workflow.py` prints four lines:

| line | what it shows |
|---|---|
| `record` | a recording handler answers each op from canned responses and keeps the trace |
| `replay` | the trace answers every op: no model, no thermostat |
| `guardrail` | the model answers 45°C, the `Gated` check fails, and the workflow returns a `Repair` before it reaches the thermostat |
| `drift` | a changed program meets the old trace, and replay raises `ReplayMismatch` |

## What replay guarantees

Replay matches ops **by name and order**. A program that yields a different sequence of ops fails
with `ReplayMismatch`. A changed prompt, tool argument or result type keeps the same names and
order, so it replays silently and is served the recorded result. That is what lets a durable run
resume after a deploy, and it is why a test pins what a workflow yields rather than only what it
returns.

`resolve` returns `Repair(reason)` when a check fails; the workflow decides what to do with it:
return it, as above, or re-prompt with the reason.

## Handlers and engines

| handler | what the ops mean |
|---|---|
| `RecordingHandler` | canned answers, no I/O; the run is recorded as a trace, and an unmet wait parks the run |
| `ReplayHandler` | the trace answers every op |
| `DurableHandler` | each op is a checkpointed step of a task, on either engine below |

| engine | for |
|---|---|
| [Absurd](https://github.com/earendil-works/absurd) on Postgres | many workers over one queue (vendored and pinned in `infra/absurd`) |
| embedded SQLite (`effective.sqlite`) | one process and one file, with no service to run |

The two engines run one conformance suite through the same handler (`tests/_conformance.py`), and
the durable tests crash a run at every op and resume it.

## What's in the box

All under `src/effective/` unless noted.

| capability | where |
|---|---|
| the op set and the authoring API: `step`, `ask_llm`, `call_tool`, `await_event`, `await_until`, `sleep_until`, `append_ledger` | `ops.py`, `domain.py`, `api.py` |
| typed prompt channels: `render`, `Gated`, `Repair`, cache and role directives, skill disclosure | `channels.py`, `skills.py` |
| identity: every op key composed from a t-string by one injective grammar | `keys/` |
| concurrency: `gather`, `race`, `quorum` | `api.py`, `choice.py` |
| combinators: `recurse`, `route`, `descend`, and sandboxed code as an effect (`run_code`, on Pydantic Monty) | `combinators.py`, `code.py`, `monty.py` |
| judgments: `judge` and `select` ask typed questions from a t-string | `judgment.py`, `api.py` |
| the ReAct agent loop, with interrupts, steering and in-flight cancellation | `react.py`, `interrupts.py`, `steering.py`, `cancel.py` |
| child tasks, and a parent that hears how its child ended | `spawning.py` |
| permission: a fail-closed cascade of rules and a human, and the `govern` gate | `permission.py`, `govern.py`, `parked.py` |
| cost: metered model calls, spend caps that survive a crash, a cache | `cost.py`, `budget.py`, `spend.py`, `cache.py` |
| telemetry: OTLP/JSON spans under the OpenTelemetry GenAI conventions, joined to the run by key | `telemetry.py` |
| counterfactuals: fork a recorded run at an op, change one answer, replay the rest | `fork.py`, `counterfactual.py` |
| optimization: `improve` (reflective prompt evolution) over a Pareto frontier of cost, quality and latency | `improve.py`, `pareto.py`, and the benches in `src/agent/` |
| the read side: a run as a graph, cards, a dashboard, a terminal viewer | `graphview.py`, `cards/`, `dashboard.py`, `runview.py`, `src/tui/` |
| interpreters that answer ops: OpenAI, the `claude -p` and `codex exec` CLIs, a judge service, a shell, the web | `interpreters/` |

`docs/effective-101.md` takes these in order, and `wiki/concepts/architecture.md` maps where each
lives.

## Examples

| run | what it is | needs |
|---|---|---|
| `uv run python examples/first_workflow.py` | the first workflow above | nothing |
| `uv run examples/smol_agent.py <url>` | a whole agent in a few lines, ported from Thomas Schranz's `smol.clj` | an OpenAI-compatible Responses endpoint |
| `uv run python examples/smol_durable.py` | the same house agent, durable and interruptible, with a judge guarding the door (`examples/smol_door.py`) | nothing: offline without keys; `OPENAI_API_KEY` and `JEV_API_KEY` for a real model and judge |
| `uv run python -m examples.coder <dir> "<task>"` | a small coding agent whose test suite decides when the work is done (`src/examples/coder/README.md`) | `OPENAI_API_KEY`, Podman |
| `uv run python -m examples.deep_research "<question>" cell="..."` | a frontier search over web leads, stopped when two hosts agree on every cell (`src/examples/deep_research/README.md`) | `BRAVE_SEARCH_API_KEY`, `JEV_API_KEY`, the `claude` CLI |
| `just tui-demo`, `just dashboard` | the terminal viewer and the run dashboard over a demo run | the `tui` extra for the viewer |

## Requirements

| need | for |
|---|---|
| [uv](https://docs.astral.sh/uv/) and git | everything; uv installs the pinned Python 3.14 (`.python-version`) |
| [just](https://github.com/casey/just) | the recipes; `just` alone lists them |
| rootless [Podman](https://podman.io/) | the durable lane (a digest-pinned Postgres), the coder's sandbox, the sandboxed formal checks |
| [elan](https://github.com/leanprover/elan) (Lean), node and npm, a JDK 17+ | the formal gates only (`just formal-setup`, `just formal`) |

Extras: `tui` (the terminal viewer: textual, rich), `judge` (the Jev judge interpreter),
`bridge` (redis, for the arrivals bridge).

## The gates

```bash
just check        # preflight, lint (ruff, the effect-boundary rules, ty), docs, wiki, keys, tests
just test-fast    # the infra-free tests alone
just pgt-up       # start and bootstrap the test Postgres under Podman
just pgt-test     # the durable lane: both engines, crash at every op, suspend and resume
just cov          # coverage, gated at 90%
just formal       # Lean proofs and the Quint typecheck
just pgt-down
```

Without a test Postgres, `just check` skips the Postgres-backed tests and runs everything else. To
run the test Postgres on another port, set `PGTEST_PORT` for every recipe, or point `DATABASE_URL`
at a database of your own.

## Layout

```
src/effective/        the substrate: ops, the authoring API, handlers, channels, keys, layers,
                      combinators, the SQLite engine, the ledger
src/effective/interpreters/   what answers a domain op
src/agent/            evaluation: bench harnesses, scoring, subagent runtimes
src/examples/         two packaged example agents: coder, deep_research
src/tui/              a terminal view over a durable run
examples/             single-file examples: first_workflow, smol_*, demos
tests/                the suite, including cross-engine conformance
formal/               Lean proofs of the pure facts, Quint models of the interleavings
infra/                vendored and pinned dependencies, and container recipes
migrations/           Alembic migrations for the ledger
scripts/              the gate scripts and database setup
docs/                 the first workflow, the concepts, the design note, the ADRs
wiki/                 one concept a page; wiki/index.md catalogs it and indexes the ADRs
```

`docs/README.md` gives the reading order. `CLAUDE.md` is the contributor guide, for people and
coding agents alike: the invariants, the code-prose rules, and how to run things.

## References

Effective builds on Recursive Language Models, GEPA, ReAct and other recent work, and vendors
Absurd and tdom. `wiki/references.md` cites each one, with background reading on AI agents.

## License

MIT. See `LICENSE`. Copyright (c) 2026 Jim Baker.
