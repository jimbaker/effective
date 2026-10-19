# Context as seams

Context engineering decides what a model sees on each call. In Effective each region of that
context is a seam: a named, typed point a workflow composes and a deployment can tune. A prompt is
a t-string, and its interpolations are typed channels: inputs render into the context, and output
channels declare the schema the answer parses back into
([ADR-0001](../../docs/adr/0001-channel-processor.md), [`src/effective/channels.py`](../../src/effective/channels.py)).

| seam | what it is | where |
|---|---|---|
| prompt | a `Prompt[S]` over a t-string; each interpolation is a channel, and a `role=`, `cache`, `nocache` or `data` directive in its format spec places it | [`src/effective/channels.py`](../../src/effective/channels.py) |
| cache order | the stable prefix first and the volatile part last, checked at render | `CacheOrderError` in [`src/effective/channels.py`](../../src/effective/channels.py) |
| content cache | an op asked once is answered from a store when it is asked again | `Cache`, `op_digest` in [`src/effective/cache.py`](../../src/effective/cache.py) |
| skills | a catalog in the cached prefix; a disclosed body as the volatile tail | `SkillRegistry` in [`src/effective/skills.py`](../../src/effective/skills.py) |
| skill hoisting | one activation above a fan-out, its pin shared by every branch | `hoisted` in [`src/effective/combinators.py`](../../src/effective/combinators.py) |
| code | code the model or a skill supplies runs as an effect in a sandbox | `run_code` in [`src/effective/code.py`](../../src/effective/code.py), `MontyEngine` in [`src/effective/monty.py`](../../src/effective/monty.py) |
| answer | a typed run's answer contract: the schema every decide asks for, and a tail naming its fields, the rendered contract and the last refusal | `run_agent(answer=...)` in [`src/effective/react.py`](../../src/effective/react.py) |
| tools | a stable catalog in the prefix; a per-turn allowlist constrains selection | [ADR-0004](../../docs/adr/0004-react-loop-driver.md) D5, [`src/agent/bracket.py`](../../src/agent/bracket.py) |
| memory | open: a run's record is durable, and nothing carries it into the next run | [ADR-0002](../../docs/adr/0002-harness-layer-stack.md), open questions |
| MCP | an MCP server's tools would reach the loop through an interpreter for `CallTool`; nothing is built | none |

## The cache order

A provider reuses work across calls only for a prefix the calls share, so the order of the
context is a cost decision. Within a role block no cached segment may follow a volatile one:
`render` raises `CacheOrderError` when one does, and `effective.lint --channels` flags the
single-literal case before anything runs. The check sits at render because cache flags compose
through nested templates. `effective.channels` emits a provider-neutral `Message(role, content,
cache)`, and the interpreter maps `cache` to its provider's cache control; `messages_to_openai`
drops it, because that provider caches by prefix.

The content cache is a second, independent layer. `op_digest` keys an `AskLLM`, `Judge` or
`CallTool` by its content, the schema it answers in, and the identity of what answers it, such as
a model. A `Message`'s `cache` flag is a provider hint and stays out of the key. A hit reports
`Usage()`, and its entry keeps what the answer cost when it was first asked. `FileStore` is the
store that exists.

## A typed answer

`run_agent(answer=contract)` types a run's answer. The contract is a function returning
`Prompt[M]`, and each of its parameters binds the last result of the one declared tool producing
that type, so a `Gated` channel can check the answer against the evidence the run gathered; a
parameter's default stands in until its tool answers. Every decide receives `M`'s schema and a
volatile tail after the transcript: the answer's fields, the rendered contract once its evidence
binds, and the last refusal on a repair turn. The transcript never keeps the tail, so compaction
and the cached prefix never see it. A refused answer is decided again at the same level, as a
numbered repeat of the turn's op that compacts nothing, until `repairs` run out and the run ends
`Exhausted`, which `unwrap` raises as a climbing refusal. A redirect on a repair turn carries the
refusal to the next level. The provider's answer is recorded raw and resolved by
`Prompt.resolve`, live and on replay alike.

| transport | the answer |
|---|---|
| the Responses caller | shaped to the schema by the provider, read back as raw JSON |
| a caller whose answer is a string | the answer's JSON inside the string, which the interpreter asks for |
| a CLI | the schema inside the one it hands the CLI |

## Skills

A skill follows the Agent Skills `SKILL.md` layout ([references](../references.md)).
`SkillRegistry.index()` is the catalog, one name and description per skill, and it belongs in the
cached prefix. A `Skill` node discloses one body as the volatile tail, so everything before it
stays warm. `activate_skill` records a `Pin` carrying the body, the pack's scripts and a tree hash
of the skill directory, so the record says which version of a skill informed a decision. A body
that discloses itself, directly or through another skill, raises `SkillCycleError` at render.

Activation does not depend on any value the workflow computes, so `hoisted` activates each skill
once above a `gather` and threads the same recorded pin into every branch.

## Code and files

`run_code` runs code as a sequence of checkpointed segments. Each host call the code makes that
changes the world surfaces as its own `CallTool`, recorded and routed through the permission
cascade, and a denial re-enters the sandbox as `PermissionError`. The default engine is Pydantic
Monty, which gives code no ambient authority: `open()`, clocks and entropy surface as host calls.
The confinement is by capability only. The worker is an ordinary subprocess with the host's user,
filesystem and network ([ADR-0014](../../docs/adr/0014-rlm-combinators-run-code.md)).

The coding example holds the project as a `dict[str, str]` and never touches it on disk. Its
`read`, `edit` and `write` tools act on that state, and `bash` runs each command in a fresh
container with no network over a copy of the project, discarding the files it changes
([`src/examples/coder/README.md`](../../src/examples/coder/README.md)). Command-line tools reach the
example through that container, where another agent might reach them through an MCP server.

## Memory

A run's record is durable and forkable: checkpoints re-serve it and the ledger keeps what it
decided ([concepts/tapes](tapes.md)). Carrying what one run learned into the next has no design.
The choice left open is recall as a yielded, recorded op, which puts it in the trajectory and
under replay, or a layer that hydrates the prompt, which keeps it out of both. Either way a write
to memory changes what the agent knows, so it belongs at the gate, where a policy can park or
refuse it ([concepts/authority](authority.md)).

## See also

[concepts/flatten](flatten.md) for why a prompt stays a `Template` until a processor renders it,
[concepts/template-shape](template-shape.md) for parsing a template once per shape, and
[concepts/tapes](tapes.md) for the record a run leaves.
