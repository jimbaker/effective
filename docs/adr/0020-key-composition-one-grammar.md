# ADR-0020: One grammar for identity: structurally-aware key composition

- **Date:** 2026-07-24
- **Status:** Accepted. Built: `compose_key` as the sole producer over a parsed grammar
  (`src/effective/keys/`), the marker types, the static shape registry with its source map
  (`effective.lint --key-registry`, `KeyMap.explain`, `KeyMap.project`), the authority scopes, and
  the identity lints in `just lint`. Not built: a Lean statement of the serialization law, which
  [`formal/lean/Effective/Keys.lean`](../../formal/lean/Effective/Keys.lean) carries as the named assumption (A-serialize).
- **Extends:** [ADR-0001](0001-channel-processor.md), whose t-string discipline on the data axis this applies to the identity
  axis.

## Context

Every durable name in Effective is a string: a checkpoint name, a wire event name, a ledger
`event_id`. Each is an identity, and a collision in any of them is silent:

| namespace | a collision means |
|---|---|
| checkpoint keys | a step reuses another step's checkpoint |
| step names | the same, since a step's name is its checkpoint name |
| wire event names | Absurd delivers by name, first-emit-wins across the queue, so one answer resolves two parks |
| ledger `event_id`s | `UNIQUE` plus `ON CONFLICT DO NOTHING` drops the second row with no error |
| scope paths | a frame boundary is forged and two scopes share keys |

Spelled by hand in f-strings, with only the checkpoint keys going through one producer, these five
namespaces carry two bug classes.

**Immediate flattening.** Collapsing a `Template` to a string at once, asking only "static or
hole?", destroys the delimiter/data distinction, and escaping then patches over the loss. It is the
Bobby Tables antipattern, the same in SQL, prompts and keys ([`wiki/concepts/flatten.md`](../../wiki/concepts/flatten.md)). PEP 750
builds the protection in: a `Template` has no `__str__`, so every flatten is an explicit act.

**Key collision.** Three counterexamples set the design's constraints:

| counterexample | what it rules out |
|---|---|
| `t"done:{item}"` with `"m1:r2"` and `t"done:{item}:{rev}"` with `("m1", "r2")` both give `done:m1:r2` | position-awareness without knowing which template produced a string: two shapes sharing a tag, differing in arity |
| `tag:{a}:{b}` with `("m1", "x:r2")` and `("m1:x", "r2")` both give `tag:m1:x:r2` | knowing the arity alone: a count says how many fields, not where they break |
| `t"x:{a}{b}"` with `("p", "q")` and `("pq", "")` both give `x:pq` | adjacent holes, which no escaping can split |

One fact made a better design available: `Interpolation.expression` carries the author's own
identifier, so a template knows the name of every field it fills.

## Decision

### 1. One producer, one grammar

`compose_key(template) -> Key` ([`src/effective/keys/processor.py`](../../src/effective/keys/processor.py)) is the sole producer of a
durable identity. The namespaces differ in policy and share one grammar, so there is one composer.
The grammar is parsed, and the composer fills a parsed skeleton (`grammar.parse_skeleton`) with
typed values:

```
key        := term [';' term]*
term       := tag [':' coordinate [',' coordinate]*]
coordinate := atom ['/' atom]*
```

Each metacharacter has one meaning, and `grammar.METACHARACTERS` is the closed set an atom may not
contain:

| byte | means |
|---|---|
| `;` | sequence: a frame, a wrapper, the arm, the op, from outermost to innermost |
| `:` | separates a term's tag from its coordinates |
| `,` | arity: separates coordinates |
| `/` | path: separates the atoms of one coordinate (`artifact:{kind}/{subtype},{digest}`) |
| `#` | occurrence: a suffix naming the n-th ask of one identity (§7) |
| `*` | a coordinate a projection dropped (§9); no producer may write it |

Because an atom cannot contain a separator, **the parse needs no registry**: arity is countable
from the bytes, and every consumer reads a key with the same parser (`grammar.parse`).

### 2. Every hole is delimiter-free by type, and nothing is escaped

`compose_key` refuses any interpolation whose type could carry a separator. What it accepts:

| value | why it is safe |
|---|---|
| `Tag`, `AuthorityTag` | a namespace, in a term's leading position |
| `Segment` and its role subclasses (`Name`, `Run`, `Subject`, `Index`, `Ordinal`) | a `str` subclass that refuses every metacharacter, and the empty string, at construction |
| `int` (not `bool`) | has no separator by type |
| `Key` | a finished composition, spliced as its own terms after a `;` |

The terminal hole has no exemption. A value that seems to want a delimiter is one of three things,
and each has a spelling: a nested key (splice it after `;`), a path (compose its atoms with `/`), or
a foreign value whose charset nobody guarantees, which is digested to one atom
(`handlers.base.digest_atom`). Validation happens once, where the value is created, and `ty` checks
the obligation at every boundary that takes a `Segment`.

### 3. The composer's checks are structural

At compose time `compose_key` requires a leading tag (a non-empty static or a `Tag` interpolation),
refuses two adjacent interpolations, and refuses any hole outside the table in §2. The format-spec
slot belongs to the key language: it carries two directives, `default=` (an optional coordinate's
fallback) and `domain=` (what a splice admits), and any other spec is refused. A spec the composer
silently dropped would mint bytes the template does not say: `t"n:{n:03d}"` would compose `n:7`.

### 4. The registry is built by the lint, and it is the source map

`effective.lint --key-registry` walks every `compose_key` template statically and writes
`build/key-registry.json` (`just key-registry`). The map is derived and gitignored. A static scan
gets `file:line` by construction, costs nothing at runtime, and fails the gate when two variants of
one namespace cannot be told apart. The same artifact serves three jobs: enforcement, the decode
surface `KeyMap.explain(key)` (named fields, the producing line, any frames and occurrence), and
operator diagnosis, which starts from a key found in a live run.

**A namespace is a tagged union.** A key is a serialized constructor application: its leading tag
is the constructor, and each registered `Shape` is one variant's signature. One tag may own several
variants when `registry.separated` proves their languages disjoint from structure alone:

| ground | example |
|---|---|
| different literal tags at some term | `race:{r};choice` against `race:{r};endings` |
| disjoint arity ranges at some term, counted from the bytes | `code:seg,{n},{name}` (three coordinates) against `code;seg:{n};{name}` (none) |
| a literal discriminator at the same coordinate | `code:seg,{n},{name}` against `code:action,{j},{name};tool:{tool}` |
| different term counts | two variants that agree term by term until the shorter one ends |

A hole never discriminates, a splice absorbs every remaining term and stops the walk, and an
optional coordinate (one with `default=`) widens a term's arity to a range. An unproven pair is
refused. `skill:{name},activate` and `skill:{name},refresh,{n}` ([`src/effective/skills.py`](../../src/effective/skills.py)) and the
four `code:` variants ([`src/effective/code.py`](../../src/effective/code.py)) are unions in the tree.

Decodability and readability are one requirement. An escaped key can still be decoded by machine,
but `review%3Am1` defeats the glance, and the glance is what an operator has.

### 5. A `Key` is opaque

`Key` ([`src/effective/keys/grammar.py`](../../src/effective/keys/grammar.py)) holds parsed terms and is not a `str`. It has no `__str__`,
so an accidental flatten renders the repr and fails visibly at its consumer. Text comes out through
named exits: `stored()` for the durable form, `display()` for the human one. Two positions treat a
`Key` as text without asking: the composer's splice and the database adapters registered against
the type. `LedgerRow.event_id` and `AwaitEvent.name` are typed `Key`, so a hand-built string in
either is a `ty` error. `Key.parse` is the read boundary and validates; `authored_key` is the
author boundary and additionally refuses an occurrence suffix (§7).

### 6. Identity positions are linted by position

`effective.lint --key-composition` flags a name built by formatting in an identity slot: the first
argument of `step(...)`/`Step(...)` and the `idempotency_key`/`event_id` keywords, following one hop
of local binding so it reports the line that built the name. It is scoped by position because an
f-string is the correct render backend everywhere else (messages, logs, display text, and inside
`render` itself). A bare f-string in an identity, SQL or prompt position means a decision was
skipped.

Concatenating a prefix onto a name is judged by the type of the right operand:

| right operand | verdict |
|---|---|
| a frame atom | a render of an already-decided structure: `frame.frame_path`, over `scope_prefix`, which refuses an atom carrying the frame delimiter `;` |
| a finished `Key` | composition: the named exit is `Key.prefixed` |
| an untyped name | an injectivity hole, closed by typing the slot `Key` (`AwaitEvent.name`, `TaskContext.await_event`) |

`effective.lint --terminal-holes` keeps `tests/` wrapping its terminal holes in a marker. The
composer refuses a non-atom in every position at runtime, so this rule is a marker discipline and
the composer holds the safety property.

### 7. Authority namespaces declare their reach

`AuthorityTag` ([`src/effective/keys/marker.py`](../../src/effective/keys/marker.py)) is a `Tag` whose names are the authorization: an
approval, a grant, a gate's park, a counterfactual's identity. Its `scope` argument is required, so
"somebody forgot" cannot pass for a considered choice. `compose_key` carries the declared scope onto
the finished `Key.scope`, and the walk reads it:

| `Scope` | means | enforced by |
|---|---|---|
| `SETTLEMENT` | one answer settles one op-occurrence (`approve`, `govern`, `depth-grant`, `chain-grant`, `round-grant`) | `handlers.base.placing` applies `Key.occurrence` to every settlement await, in every walk, so a second ask is a second name |
| `ACCRUAL` | one answer stays in force for a declared scope on purpose (`budget-grant`, `generation-grant`, `gate-state`) | nothing: it is the named opt-out |
| `QUALIFIED` | wraps another composed key and relocates it (`fork`, `hyp`) | `--authority-scopes` requires the wrapped hole to be a `Key`, so its own namespace was checked when it was made |

The scope is declared because it is semantic. `depth-grant:{run_id},{generation}` and
`budget-grant:{run_id},{trip}` are equally injective functions of their fields, and nothing in the
template says that the first must settle one question while the second's run-wide reach is the
feature. A fourth value for state cells was considered and dropped: its only possible check could
report a near-miss and never a defect.

**The occurrence suffix.** `#N` qualifies a whole key and has one producer, `Key.occurrence`, which
is the identity at n ≤ 1, so a name asked once is byte-identical to the bare key. It matches what
the Absurd SDK and `SqliteTaskContext` already store for a repeated checkpoint name. `#` is in no
atom charset, `authored_key` refuses it in an author's name, and `KeyMap.explain` reports it as
`Explanation.occurrence`, separate from the template's fields.

**Fences.** A `Step`'s key opens with its own arm term (`step;approve:x`), so the step region is
disjoint from every substrate namespace by construction and needs no denylist. An await is
addressed by its event name, which carries no arm, so `RESERVED_AUTHORITY_TAGS` fences awaits, and
`effective.lint --authority-tags` checks it equal to the `AuthorityTag` declarations in both
directions. The set is matched as a parsed leading tag, so there is no prefix spelling to go stale.

### 8. Identity comes from content, a name, or a position, and the key agrees

| identity | key | arms |
|---|---|---|
| content | content-addressed | `StoreArtifact` |
| named | the name | `Step`, `AwaitEvent`, `AppendLedgerRow` |
| positional | assigned by the walk | `Gather`, `SleepUntil`: `op_key` raises for both |

A sleep's wake time is data, so it lives in the run's state and never in the key. `placing` mints
`sleep:{n}` from `FramePosition`, a per-frame counter shared by every walk, and publishes the name
around each drive loop's call into the layer stack. That placement matters twice: an authority
layer and the base interpretation bind the same identity, and a layer that re-forwards an op does
not advance the ordinal, because the workflow yielded once.

The rule splits by origin. A suspension the workflow authored is identified positionally by the
walk; one a layer injects carries a deterministic name from whatever injected it (the permission
`AwaitEvent` in [ADR-0002](0002-harness-layer-stack.md)'s "Which ops reach the op-layer stack").

Frames are leading terms. `scope_prefix` renders a `scoped(...)` atom, `gather_prefix(g, i)` and
`race_prefix(r, i)` render the substrate's own branch frames through `compose_key`, and
`api.qualified_event_name` composes the same frames for an emitter, so the name an emitter wakes
and the name a park binds come from one composition. `monitor` is reserved in `frame.ARM_TAGS` for
an arrivals monitor and names no op today.

### 9. A projection is a second kind of key

`KeyMap.project(key, drop=…)` answers which coordinates of a stored key a view ignores. It returns a
`ProjectedKey`, a sibling of `Key` that has `display()` and no `stored()`:

| | `Key` | `ProjectedKey` |
|---|---|---|
| minted by | `compose_key`, `Key.parse` | `KeyMap.project` only |
| renders | `stored()`, `display()` | `display()` only |
| `parse` | accepts | refuses: `*` is a metacharacter |
| durable | `LedgerRow.event_id: Key` | nothing stores one |

A subclass of `Key` would satisfy every `isinstance` check that decides what may be written, which
is the property the type exists to deny. A dropped coordinate is replaced by `*`, so a projection
keeps its arity and two shapes of different arity cannot collide. Projecting a projection
re-projects `source` under the union of the drop sets, so
`project(project(k, A), B) == project(k, A | B)` holds by construction. `claimed` says whether the
map read the innermost term, the one carrying the op's identity, and `fold_cycles` reports the
rest as `RunGraph.unclaimed`.

The roles a projection drops by are declared at the mint, since the bytes cannot say them:

| role | a fold |
|---|---|
| `Name` | keeps it: distinct positions |
| `Run` | keeps it; a cross-run comparison drops it |
| `Subject` | keeps it: the domain's value |
| `Index` | drops it: re-executions of one position |
| `Ordinal` | keeps it: distinct positions of one kind, such as two straight-line sleeps |

`gather:{g}` is declared `Index`, though `g` numbers distinct gathers, because that preserves every
pinned tape. The witness that would overturn it is two straight-line gathers in one frame.

A folded key is a function of the recorded name and of `build/key-registry.json`, so an index
built over folded keys is rebuilt when a mint site's declaration changes ([ADR-0022](0022-dashboard-projections-the-read-side.md) §9c).

## Consequences

**Good.** Escaping is gone, replaced by a type obligation checked statically. Keys are readable and
decodable by the same parser that composed them. The two bug classes have gates. The source map
turns a key found in production into a line of code. The identity axis has the safe-composition
discipline the data axis has had since [ADR-0001](0001-channel-processor.md).

**Costs.** Every substrate name producer takes a marker type. The registry is lint machinery plus a
derived artifact, and `KeyMap.load` needs it present: an installed wheel ships no `build/` tree,
so a caller outside a checkout builds a map with `KeyMap.from_shapes`.

**Risk accepted.** The registry restates each template's tag. It is generated from the templates,
the way `--authority-tags` derives its set; if it is ever maintained by hand, this decision has
failed.

**What the gates do not prove.** The registry proves string-level disjointness between variants.
It cannot prove a shape's fields are a complete coordinate of the question being asked, which is
why scope is declared (§7). `--authority-scopes` reads same-module constants only, so a tag
declared in one module and composed in another is not resolved; every authority tag in the tree is
declared and composed in one module. `Step.name` is still a `str`, so an author's step name is
checked at `step_key` and never registered. Serialization injectivity is held by construction and
by sampling ([`tests/test_op_key_injectivity.py`](../../tests/test_op_key_injectivity.py)), and Lean proves it only for the structured
encoding.

## Alternatives considered

| alternative | why it lost |
|---|---|
| structured identity on the wire (a tuple, no in-band delimiter) | checkpoint names and `event_id`s are persisted strings and the Absurd SDK is co-versioned; the churn crosses the wire format, both engines and existing data. In-process a `Key` does hold parsed terms; the wire keeps a flat, typeable rendering |
| four per-namespace composers | the namespaces share one grammar and differ in policy, so four producers quadruple the surface the injectivity argument covers |
| escape every hole | defeats the glance, and leaves the hand-spelled sites outside the producer |
| position-awareness without a registry | refuted by the first counterexample in Context |
| arity on the wire (`tag:5:…`) | a count does not say where fields break (second counterexample), and field names and the producing site live only in the source. The `,` separator now makes arity countable from the bytes anyway |
| a denylist of reserved step prefixes | the same shape as the antipattern it guards; the arm term makes the step region disjoint by construction |
| a `compose_path` sibling for `/` | `/` is a metacharacter of the one grammar, admitted in a static and refused in a value exactly as `:` is |
