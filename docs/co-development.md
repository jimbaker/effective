# Building with a coding agent

Effective was built with coding agents, and a workflow on it is meant to be built the same way. A
coding agent such as Claude Code or Codex writes the code. The gates decide whether the code is
right: tests, the type checker, lints for the rules a reader cannot see, and reviews. A person
merges what passes.

That split works because an Effective workflow can be checked offline. The workflow is a generator
that performs no I/O, so an agent can drive it with canned answers and observe every op it yields,
with no key and no spend. A run's recording is a fixture the agent can replay as often as it likes,
and a replay finds a change in which ops the workflow yields, compared by key.

## The three stages

| stage | what the agent does | what Effective supplies | the gate that closes it |
|---|---|---|---|
| zero to one | writes the workflow as a generator against scripted answers: one op per thing that touches the world, a `t"..."` prompt with typed channels, pure code between yields | `RecordingHandler` and `ReplayHandler`, `render` with `Field` and `Gated` | a test that drives the workflow with canned answers and pins its result and its ops; `just lint`, which refuses I/O between yields and a bare `yield` |
| productize | makes it reusable, composable, durable and replayable: factors steps into workflows composed by `yield from` or spawned as child tasks, and runs it as a task on the embedded SQLite engine | the combinators, `DurableHandler`, the SQLite engine, keys composed by one grammar | a test that crashes the run after each op and resumes it to the same result, and a replay of its recording; in this repository, `tests/_conformance.py`'s `at_every_op` and `just check` |
| deploy | points the same workflow at Postgres, with workers draining one queue | `effective.engines.open(url)`, which opens SQLite or Absurd on Postgres by URL; spend caps; OTLP spans | the same tests against the engine it will run on; in this repository, `just pgt-up` then `just pgt-test` |

[`docs/first-workflow.md`](first-workflow.md) is the first stage, done. [`wiki/concepts/testing.md`](../wiki/concepts/testing.md)
lists the instruments behind the second stage's gate. A walkthrough from a fresh managed Postgres
to a running worker is not written yet.

## What the gates catch that the agent cannot see

An agent reads the code in front of it. The gates check the rules that live elsewhere:

| rule | what breaks it | the gate |
|---|---|---|
| a workflow is deterministic between yields | an I/O or `random` import, or the common spellings of a clock call, in a registered workflow file | `just lint`; `open`, `print` and an aliased clock pass it, as [`docs/first-workflow.md`](first-workflow.md) says |
| every key is composed by one grammar | a key built by `+` concatenation, or an f-string under `src/`; an op name built from a value can still take another op's key | `just lint`, and the f-string gate |
| an op's recorded result fits its type | a changed result type that an earlier recording does not fit | a durable resume, which refuses the recorded value; an in-memory replay serves it unchecked, and serves the recorded result for a changed prompt or argument too |
| a run survives a crash anywhere | a side effect outside an op, which runs again on resume | the crash-at-every-op tests, where the test observes the side effect |
| the code is typed as it claims | an unchecked `Any` crossing a seam | `uvx ty check` |

A failing gate gives the agent a reproduction, and the agent fixes it. When the gate is right, a
change that turns it green without answering its reason is a defect. Respelling an f-string as a
join to pass the f-string gate is the common case.

## The reader is a person and an agent

Every page, name and error message here has two readers, and the agent acts on what it reads. So a
defect that a person would muddle through is still a defect:

| surface | what serves both readers |
|---|---|
| a name | one that teaches the seam: `govern` parks and `serve` never does, so the pair carries its composition law ([ADR-0019](adr/0019-govern-serve-composition-combinators.md)) |
| a named role | one that appears in a signature or an error; a `Protocol` that types nothing points an agent at nothing |
| an error | it names what the developer wrote and the fix, and fails at the seam with a named exception, before a checkpoint is half written |
| a document | it survives an agent's grep: a page naming a function the code does not have is a correctness bug, since the agent trusts it |
| a rule | a type or a guard that rejects the mistake, in preference to a convention a reader must remember |

A guard is a claim about an adversary, so name the adversary before believing it:

| adversary | the test | what passing proves |
|---|---|---|
| the accident | revert a call site to the natural mistake, such as an f-string where a composed key belongs | the guard catches a slip |
| the forge | construct the typed value directly from a string, and pass it | the guard catches intent |
| the launderer | find what already mints the type from untyped input, and check whether it parses first | the guard covers anything at all |

Write down in the guard's docstring which adversary it stops and which it does not. The same
question applies to a gate: ask what it scans before trusting what it rejects.

## What the agent proposes and the person decides

An agent writes most of the code. A name, a seam or a piece of vocabulary is a decision code and
agents will depend on, and a person rules on it:

| step | who | what |
|---|---|---|
| census | the agent | count each candidate word's uses across the tree before proposing one; a collision is measured, never recalled |
| borrow | the agent | take the common name from a named field, so a reader can look it up; at a domain seam, the domain's word (a *deduction* to a finance reader), and for the mechanism, the project's |
| find the axis | the agent | when two concepts want one word, name what distinguishes them, and the words follow |
| ask whether it should exist | the agent | a naming question is a cheap place to find a thing nothing needs |
| lay out the options | the agent | a table of candidates, each with what it buys and costs: legibility to each reader, accuracy, how far a rename reaches, whether it stays exact at the second case |
| rule | the person | take the option past which more legibility costs more than it buys |

The round trip is cheap next to shipping a wrong name, which is expensive to change once code and
agents depend on it.

## How this repository works with agents

A coding agent proposes each change and organizes its own work, including splitting a task among
subagents. A change merges only after `just check` passes and a reviewer that did not write it has
tried to break it. Changes to durable, concurrency or invariant code get a second reviewer from
another model vendor. A person merges.

A contributor's agent may do what a contributor may do, and its work passes the same gates before a
human merges it. `AGENTS.md` is the guide for both, and Claude Code and Codex each read it at the
start of a session: the layout, the recipes, the invariants and the rules for code prose.
