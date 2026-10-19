# Building with a coding agent

Effective is for building repeatable agentic workflows with a highly capable coding agent such as
Claude Code or Codex. The usual starting point is a one-off: a task the agent has already done once
in a session. The session's telemetry (its prompts, its tool calls and what they returned) shows the
steps, and the agent turns them into a workflow that runs the same way each time. The agent writes
the workflow and its tests. Checks the agent can run itself decide whether the code is right. A
person decides what the workflow is allowed to do.

That split works because an Effective workflow can be checked offline. The workflow is a generator
that performs no I/O, so an agent can drive it with canned answers and observe every op it yields,
with no key and no spend. A run's recording is a fixture the agent can replay as often as it likes,
and a replay finds a change in which ops the workflow yields, compared by key.

## Everything runs where the agent runs

Effective needs no container and no service of its own. The SQLite engine is one file, `run_code`
runs on Pydantic Monty in process, and a workflow's model and tool calls can be answered with canned
results or a replayed recording. So an agent in a sandbox, such as Claude Cowork, can build, run and
check a workflow end to end, and a nontechnical user working with it sets nothing up. SQLite serves
in production as well, for a workflow that runs in one process: deep research, a search over web
leads, needs only its API keys ([`src/examples/deep_research/README.md`](../src/examples/deep_research/README.md)).

## The three stages

| stage | what the agent does | what Effective supplies | the check that closes it |
|---|---|---|---|
| zero to one | turns a one-off's steps into a generator, run against scripted answers taken from the one-off's tool results: one op per thing that touches the world, a `t"..."` prompt with typed channels, pure code between yields | `RecordingHandler` and `ReplayHandler`, `render` with `Field` and `Gated` | a test that drives the workflow with canned answers and pins its result and its ops; `python -m effective.lint` on the workflow file, which refuses I/O between yields and a bare `yield` |
| productize | makes it reusable, composable, durable and replayable: factors steps into workflows composed by `yield from` or spawned as child tasks, and runs it as a task on the embedded SQLite engine | the combinators, `DurableHandler`, the SQLite engine, keys composed by one grammar | a test that crashes the run after each op and resumes it to the same result, and a replay of its recording; Effective's own suite does this in [`tests/_conformance.py`](../tests/_conformance.py), and no importable helper ships yet |
| deploy | points the same workflow at the engine it will run on: SQLite for one process, or Absurd on Postgres with workers draining one queue | `effective.engines.open(url)`, which opens SQLite or Absurd on Postgres by URL; spend caps; OTLP spans | the same tests against the engine it will run on |

[`docs/first-workflow.md`](first-workflow.md) is the first stage, done. [`wiki/concepts/testing.md`](../wiki/concepts/testing.md)
lists the instruments behind the second stage's check. A walkthrough from a fresh managed Postgres
to a running worker is not written yet.

## What the checks catch that the agent cannot see

An agent reads the code in front of it. The checks hold the rules that live elsewhere:

| rule | what breaks it | the check |
|---|---|---|
| a workflow is deterministic between yields | an I/O or `random` import, or the common spellings of a clock call | `python -m effective.lint`; `open`, `print` and an aliased clock pass it, as [`docs/first-workflow.md`](first-workflow.md) says |
| an op's recorded result fits its type | a changed result type that an earlier recording does not fit | a durable resume, which refuses the recorded value; an in-memory replay serves it unchecked, and serves the recorded result for a changed prompt or argument too |
| a run survives a crash anywhere | a side effect outside an op, which runs again on resume | a crash-after-each-op test that observes the side effect |
| the code is typed as it claims | an unchecked `Any` crossing a seam | a type checker such as `ty` |

A failing check gives the agent a reproduction, and the agent fixes it. When the check is right, a
change that turns it green without answering its reason is a defect, such as a test re-pinned to
whatever the code now prints.

## What the agent writes and what the person decides

The agent writes the code. The person decides what the workflow may do, and each of those
decisions is a value in the workflow that an agent can read and should not change on its own:

| decision | where it lives | why it is the person's |
|---|---|---|
| what the workflow may commit | the tools it can call, and a `govern` policy that parks a call for approval | an approval is authority, and the agent that does the work cannot grant it |
| what a run may spend | budgets under `govern` | a ceiling the worker can raise is no ceiling |
| what an answer must satisfy | a `Gated` channel's predicate | the predicate is a business rule stated as code, such as "the citation names a passage" |
| an op's name | the first argument of each op | the name is the op's key; a run in flight finds no record under a new name and performs the op again |

A person can review a workflow's behavior without reading all of its code. A recording lists every
op a run took and what it returned, and a fork (`fork_at`, `run_fork`) changes one recorded result
and runs the rest live, so "what does the workflow do if the search returns nothing" is a run the
agent produces and the person reads.

## A guardrail has an adversary

A `Gated` channel is a claim about what it stops. Name the adversary before trusting it:

| adversary | the test | what passing proves |
|---|---|---|
| the slip | a scripted answer with the natural mistake, such as a citation of a passage that does not exist | the guardrail catches the model's error |
| the injection | a passage or tool result that tells the model to break the rule | the guardrail holds when its input is hostile |
| the side door | another path to the same commitment, such as a tool that writes without passing the gated answer | the guardrail covers the commitment at all |

The `Repair` a channel returns carries a reason, and a re-prompt can show it to the model, so write
the reason for the model to act on: "cite a passage by its number" works where "invalid" does not.

## Telling the agent

Put the rules a reader of the code cannot see in the project's `AGENTS.md`, which Claude Code and
Codex both read: which files are workflows, how to run the offline tests, and which of the values
above the agent must not change without asking.
