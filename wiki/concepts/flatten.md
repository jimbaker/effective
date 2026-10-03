# Flatten at the boundary, not before it

`effective/channels.py`, `examples/first_workflow.py`, `effective/keys/processor.py`, the psycopg SQL
boundary, `infra/tdom`.

**A `Template` has no `__str__`.** PEP 750 designed that in: `str(template)` and `f"{template}"`
both give the repr, so you cannot flatten by accident and every flatten is an explicit act. Code
that flattens immediately has **opted out of the protection the language handed it**.

## The antipattern, and it is one shape in three grammars

Accepting a `Template` and collapsing it to a string right away, asking only "static or hole?",
never *which* hole, how many, what is beside it, or what the author wrote, destroys the
delimiter/data distinction. Escaping then becomes a patch over the loss rather than a property of
the structure. That is Bobby Tables, and it is the same defect wherever it appears:

| grammar | the injection it invites | its processor |
|---|---|---|
| SQL | Bobby Tables proper | the psycopg t-string boundary |
| prompts | prompt injection | `effective.channels` |
| keys | delimiter forging, checkpoint and approval aliasing | `compose_key` |
| HTML / SVG | markup injection | `tdom.html` |
| web search queries | operator forging: a value carrying `-site:` or `OR` rewrites the search | `effective.query.search_query` |
| markdown | a `\|` splits a table cell, a `*` opens emphasis, a newline then `#` starts a heading | `effective.markdown`, with `table` aligned so it reads unrendered |
| URL paths, data URIs | none | **none yet, which is a gap rather than a licence** |

`compose_key` flattened immediately for months (`"".join(map(_key_segment, template))`, a
hand-rolled flatten), which is how it shipped an injectivity hole on adjacent interpolations. A
`Template` gives you position, arity, and each hole's source `expression`; a processor that ignores
all three kept the syntax and threw the feature away.

## It is a layering rule, not "f-strings bad"

An f-string is eager format-then-concat, which is exactly right **once the structural decisions are
already made**. So f-strings belong *below* a processor as its rendering backend, and t-strings
belong *at the boundary* where the decisions still have to be made. The defect is never the
f-string; it is putting the flatten where the **decision** should be.

The backend position is a positive choice. An f-string is the fastest way Python has to build a
string, which is what a processor's hot path wants: `op_key` runs on every op. Measured here, 2M
iterations on a 3-segment key:

| | ns/op | rel |
|---|---|---|
| f-string, fixed shape | 26.6 | 1.00x |
| `str.join`, variable arity | 27.6 | 1.04x |
| `%`-format, `+` concat | 37.8 | 1.42x |
| `str.format` | 72.1 | 2.71x |

**f-string when the processor knows the shape, `str.join` when arity is variable**: they are at
parity. Reaching for `.format` in a hot render is a 2.7x tax for no gain.

**The rule:** an f-string is allowed for a t-string processor's **render backend**, and generally
nowhere else. Read a bare f-string in an identity, SQL, prompt or markup position as *"a decision
was skipped here"*, and a `Template` as *"the decision happens here."* Ask what totality asks of an
`isinstance` chain: **was a structure available, and did this flatten it?**

## It is a gate

`scripts/fstring_sweep.py --gate src`, inside `just check`. Two designated positions are named in
the script and never baselined; every other entry records the processor it wants, and the gate
fails on a **new** f-string and on a baseline entry nothing produces, so the record can only
shrink. Quote the command, not the number; the count moves.

The gate has a hole worth knowing before you trust a green: it keys on `(path, statics)` and
dedupes, so a new f-string in an already-baselined file passes. Keying the baseline per site is
open.

A second hole is the one an author falls into: the gate scans for f-strings, and a `str.join` or a
`+` that replaces a flagged one is the same flatten, invisible to it. Respelling turns the gate
green and leaves the skipped decision where it was. A search query whose page value could forge
`-site:` and a markdown table a `|` could break are two such respellings, answered by the
processors `effective.query` and `effective.markdown`. Dispose of a finding by asking which
processor wants it.

## The processor decides the structure

The rule above is **relative, not absolute**. Which structure a `Template` must carry is a fact
about what will consume it.

| processor | reads | so the right shape is |
|---|---|---|
| `compose_key` | statics, interpolations, **and each hole's `expression`** for the key registry's field names | AST construction is the whole point |
| `tdom.html` | the structure, and `expression` only to format a parse error | a nested `Template`, a list, a list of lists and a generator all render identically |

PEP 750 says both halves. It expects nesting and composition *"in preference to simple string
concatenation"*, and says *"the `expression` attribute will not be used in most template processing
code."* So the question at a t-string seam is not "is this shape clean?" but **"who processes this,
and what does it read?"**, and if nothing reads that, there is no rule to enforce.

That was learned by cost. A `--hole-control-flow` lint was built here to forbid a comprehension
inside an interpolation, on the reasoning that it destroys the hole's `expression`. It was
well-made (a ten-shape discrimination matrix, firing on the real file, red-then-green under
mutation), and the rule was wrong: it forbade a construction the PEP expects, to protect a field
the PEP says most processors do not read. Built and reverted the same day.

## The vocabulary is the PEP's, exactly

`template`, `Template`, `Interpolation`, `strings`, `interpolations`, `values`, `expression`,
`conversion`, `format_spec`. `string.templatelib`'s public surface contains **zero abbreviations**,
so `tpl` is not a clipped domain term: it is Jinja/Go/PHP jargon advertising the wrong mental
model. Foreign-neighbourhood vocabulary is the failure mode to watch, not shortness.

## The test corollary

An assertion on **rendered bytes** has flattened the structure before asking its question, exactly
as an f-string has. Three pins broke or misled this arc that way: `&quot;` against tdom's `&#34;`,
`<!doctype>` against `<!DOCTYPE>`, and `assert "<script>…" not in html`; the third passes a
renderer that **drops its input entirely**, which mutation testing confirmed. Parse and assert on
the result (`tests/_html.py`), and the assertion says what it means while the renderer stays free
to spell an escape however it likes.

## Reach for what only a t-string can do

Because a `Template` carries each hole's source expression, a composed key can be decoded back into
named fields and linked to its producing line, impossible with an f-string. When you
touch a t-string seam, ask what the structure makes newly *possible*, not just what it makes safe.

The second instance is the **prompts** row. A hole spelled
`{run.output:data}` renders between `[[ ## data run.output ## ]]` and `[[ ## end ## ]]`, labelled
with that same source expression, and the mark moves to one the content does not use. The marker
alphabet shares no metacharacter with JSON, shell or code, so nothing is escaped and a file arrives
byte for byte. The first shape tried was a tag family, which was wrong twice: angle brackets are
what the content is MADE of when an agent reads markup, and qualifying the tag marked depth by a
counter. Both are the homogeneous nesting the M1 wire ladder measured at 27 of 120 trials. Trust rides the slot rather than
the channel because it is extrinsic: one `str` is authored prose at one site and a command's output
at another. `effective.sql` draws the same line one grammar over, binding every hole as a parameter
unless its spec says identifier. Live in `src/examples/coder`.
