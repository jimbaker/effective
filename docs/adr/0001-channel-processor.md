# ADR-0001: The channel processor: a typed function on `Template`, with composition

- **Date:** 2026-06-01
- **Status:** Accepted. Built in [`src/effective/channels.py`](../../src/effective/channels.py): the `Output` marker, `render`,
  `Prompt[S]`, role/cache/data directives, the volatile-last check, nested-`Template` composition,
  the channel/signature validator and `Skill` nodes. Not built: priority-based budget pruning,
  categorical strategy seams, an auto-rendered output contract.

## Context

`effective.channels` reifies the **data axis**: a prompt is a PEP 750 `t"..."` whose
interpolations are typed I/O channels. The canonical t-string idiom (tdom's `html(t"...")`,
psycopg's `cur.execute(t"...")`) is a **function on `Template`** that walks strings and
interpolations as a DSL and builds structure, and composition is what that idiom is for: nested
sub-prompts, named fragments, cache-aware layout.

The JS/TS ecosystem, at the same fork, chose typed component trees (JSX/TSX: Priompt,
prompt-tsx) over tagged templates, because the context window is a layout problem and only a
tree is prunable. A nested `Template` is a tree too. The choice here is to adopt the processor
in its legible core: typed end to end, with turns and the optimizer on their own axes.

## Decision

The channel processor is `render(t"...", *, output=S, registry=None) -> Prompt[S]`, a pure
function over `Template`. Six parts:

| part | decision |
|---|---|
| A1 processor and composition | `render` walks the template; a nested `Template` interpolation composes |
| A2 typed result | `Prompt[S]` carries the declared output type; `resolve` returns `S \| Repair` |
| A3 turns stay on the control axis | the processor renders one turn's context; the ReAct loop and the combinators never enter a `Template` |
| A4 skills as the named composition unit | `skill(name)` splices a registry-resolved or pinned body |
| A5 the DSL tooling | a volatile-last cache check and a channel/signature validator |
| A6 per-prompt choice | a flat prompt and a composed one use the same channels, so the choice falls out of A1 to A5 with nothing to build |

Deferred, with a reserved surface each:

| deferred | trigger | reserved surface |
|---|---|---|
| categorical strategy seams for an automated optimizer | an optimizer that enumerates seam values ([ADR-0005](0005-optimizer-agent-in-the-loop.md) stage 3) | `Prompt.seams`, the `{expression: rendered text}` map |
| priority and budget-aware pruning | the first prompt that overruns its budget | none yet: channels carry no `priority` field |
| an auto-rendered output contract | categorical strategy seams | the author writes the JSON-key lines; the processor fills only `<<name>>` |

## Architecture

### Typed channel vocabulary

`Field`, `Gated`, `FormGate`, `Channel`, `Done` and `Repair` stay. The `Output` marker makes a
reusable `Annotated` type an output channel, validated through a cached `TypeAdapter`, so the
guard, schema and direction ride the type and the repair reason is Pydantic's:

```python
from typing import Annotated
from annotated_types import Interval
from effective.channels import Output

Celsius = Annotated[float, Interval(ge=10, le=30), Output]
```

`Gated` objects stay for the dynamic predicate, one whose bound is per call:
[`examples/first_workflow.py`](../../examples/first_workflow.py) gates a setpoint against a caller-supplied `ceiling`.

### The processor

`render` walks the `Template` in order and classifies each interpolation by its value:

| value | meaning |
|---|---|
| nested `Template` | recurse, inheriting role and cache; merge its channels and seams |
| `Skill` | resolve the body from its pin, else from `registry`, and walk it; the seam is keyed by the skill name |
| `Citation` (`cite(text, address)`) | quote the text in a fenced data block labeled with the record's address |
| `Done` or `Repair` | refused with `IndependenceError`: a prior resolution re-entering a render is a turn boundary |
| a `Channel` or an `Output`-annotated type | an output slot, collected under `interp.expression`, rendered as `<<name>>` |
| any other value | an input, rendered in place under its `format_spec` |

A `format_spec` is either an ordinary Python format spec or a directive: `role=system|user|assistant`,
`cache`, `nocache` and `data` (content from outside the prompt, fenced). A spec mixing the two
raises. The result is `Prompt(messages, channels, seams, output)`: messages coalesced by adjacent
`(role, cache)`, each a provider-neutral `Message(role, content, cache)`.

**Where `render` runs.** The workflow calls it between yields and hands `prompt.messages` to
`ask_llm`, which carries them in an `AskLLM`; `first_workflow.py` and `run_agent`
([`src/effective/react.py`](../../src/effective/react.py)) both do this. `render` performs no I/O, but it calls `__format__` on
the values it is handed, so the determinism boundary holds only for values that render the same
bytes on every run.

**Provider mapping.** The interpreter, never `effective.channels`, maps `Message.cache` to a
provider's cache control: `messages_to_openai` ([`src/effective/interpreters/openai.py`](../../src/effective/interpreters/openai.py)) drops it,
because that provider caches by prefix.

### `Prompt[S]`, the typed result

`Prompt.resolve(response, form_gates=())` writes each channel, runs the cross-field `FormGate`s,
then validates the dict into the declared `S`. A field, form-gate or model-level breach, and a
response that is not a mapping, each come back as a `Repair`, so all are re-promptable the same
way. `S` is read off `render(..., output=Setpoint) -> Prompt[Setpoint]` by `ty` and by an agent
alike: composition over an untyped `Template` would lose the signature, and the declared `output`
restores it.

### Composition and skills

A sub-prompt is a function returning a `Template`; interpolating it composes. A duplicate channel
name across the composed tree is `ChannelCollisionError` at render. A `Skill` is preferred over an
anonymous splice: it is named, so a reader sees what composed, and its body is disclosed as the
volatile tail after a cached catalog. `SkillRegistry` ([`src/effective/skills.py`](../../src/effective/skills.py)) loads
standard-layout skill packs and satisfies the structural `SkillResolver`; a pinned `Skill` renders
its recorded content and consults no registry, which is the form a durable run uses. A skill that
discloses itself is `SkillCycleError`.

### The DSL tooling

| check | where |
|---|---|
| volatile-last: within a role, no cached segment follows a volatile one | `render` raises `CacheOrderError`; `effective.lint --channels` flags the single-literal case statically |
| signature: the declared model's fields equal the channel names | `check_channels`, run inside `render` (`ChannelMismatchError`) and usable as a test helper |
| independence: no input is a prior `Done`/`Repair` | `render` raises `IndependenceError`; `effective.lint --channels` checks it statically |

The render-time cache check is the authoritative one, because cache flags compose dynamically
across nested templates and a static rule sees one literal.

## Consequences

**Positive**

- Composition is native: nested sub-prompts and named skills ride one recursive walk.
- The signature survives composition (`Prompt[S]`), so agent-legibility, the primary lens, holds.
- `seams` falls out of `interp.expression` with no extra authoring: a message-catalog surface
  and the optimizer's mutation surface ([ADR-0005](0005-optimizer-agent-in-the-loop.md)).
- Cache bracketing is correct by construction and checked.
- Typed channels replace per-site lambdas and hand-written repair reasons.

**Risks and their mitigations**

| risk | mitigation |
|---|---|
| legibility lost under composition | `Prompt[S]` and the signature validator; A2 is required |
| cache discipline | the volatile-last check is enforced, not advisory |
| combinatorial blow-up | categorical seams deferred; named skills preferred to arbitrary nesting |
| DSL surface creep | directives stay role, cache and data; no render-strategy DSL |
| expectations of editor tooling | the tooling validates scaffolding; prose cannot be highlighted |
| turns leaking into templates | `IndependenceError` draws the line in code |

## Invariants

- **Determinism boundary.** `render` and `resolve` do no I/O; workflows still `yield from` typed
  ops.
- **DB-free core.** `effective.channels` imports no SQLModel and no provider SDK.
- **Provider neutrality.** Cache control is an interpreter's mapping of `Message.cache`.

## Alternatives rejected

- **Typed channels with no processor.** Flat: no composition and no prunable tree. Kept inside the
  processor for flat prompts (A6).
- **The type named in `format_spec`.** Stringly typed, opaque to `ty` and to an agent.
- **Categorical strategy seams first.** Front-loads an optimizer that enumerates seams before one
  exists; the `seams` surface is reserved instead.
- **An untyped processor (the literal tdom/psycopg shape).** Loses the signature; rejected for A2.
- **Turns and combinators in the template.** Collapses the data and control axes; rejected (A3).

## Open questions

- Is the skill the strict default composition unit, or do skills and bare nested templates
  coexist as peers?
- Is a summarizing combinator ever template-side (rendering a value the parent should see), or
  always on the control axis?
- When channels gain a priority, is pruning a `render` concern or a post-pass over `Prompt`?
  Priompt and prompt-tsx put it in the renderer.

## Why `interp.expression` matters here

Seam names, the `seams` dict and the optimizer surface all come from `interp.expression`, the
attribute PEP 750 keeps for catalog and metaprogramming uses. The property that makes `t"{total}"`
debuggable makes `{"total": "<rendered>"}` an addressable, swappable seam, without wrappers and
without the author writing a name twice. `format_spec` carries the extrinsic intent, and the
deferred `Template` keeps values unstringified for caching. Those three PEP 750 properties are
the case for this path over the JSX trees JS/TS chose.
