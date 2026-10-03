# Effective: the contributor guide

For people and coding agents alike. Effective is a durable-workflow substrate built on algebraic
effects: a workflow is a plain Python generator that **yields typed op descriptions**, and a
swappable handler interprets them (recording and replay in tests; durable execution on Absurd and
Postgres, or on the embedded SQLite engine). Stack: Python 3.14, `uv`, Pydantic, SQLModel and
Alembic for the ledger, Postgres for the durable lane.

**The theme: weave together determinism and gen AI**, both in what Effective supports and in how
it is developed. *What it supports:* a model call is non-deterministic and a durable run must be
reproducible, so the substrate wraps the model in deterministic machinery: a typed op with a stable
key, a recorded result that replay re-serves, a permission cascade evaluated before the side
effect, a `Gated` channel that constrains the answer and repairs it. *How it is developed:* the
same weave, pointed at ourselves. A model proposes; gates, pinned tests, mutation checks,
cross-engine conformance and fresh-eyes review decide. That is why the discipline below is not
ceremony: *run the claim, don't read it*, *an instrument is a claim* and *ask what a gate scans*
are the development-side spelling of the product-side thesis.

**Two seams, and t-strings as the second.** Effects reify the **control axis**: what the workflow
does, as ops that are data. PEP 750 **t-strings reify the data axis** (`effective.channels`): a
prompt is a `t"..."` whose interpolations are typed I/O channels. Inputs render in; outputs
(`Field`, `Gated`) declare the schema and parse the response back. A `Gated` channel runs a
constraint and returns `Repair` to drive a bounded re-prompt: a guardrail on the data boundary.
Treat channels as first-class. Each is a named, typed, swappable seam you can tune and meter, so a
guardrail **is** a `Gated` channel and every seam is a variable with a measurable cost, quality and
latency.

## Python 3.14, t-strings, and the flatten rule

A t-string use here has to be **exemplary, not merely working**: one that compiles, passes `ty`,
and is safer than an f-string can still be a defect, because the standard is the best pattern
anyone could point at. The rules and their evidence are wiki pages:

| page | holds |
|---|---|
| `wiki/concepts/flatten.md` | a `Template` has no `__str__`, so every flatten is an explicit act; immediate flattening is the Bobby Tables antipattern in four grammars; an f-string is right in one position, a processor's render backend. `scripts/fstring_sweep.py --gate src` enforces it |
| `wiki/concepts/python-314.md` | the 3.14 surface in use; syntax that looks wrong here is a feature until the changelog says otherwise |
| `wiki/concepts/enforcer-domain.md` | a gate is bounded by what it scans |

Exemplars to imitate: `src/effective/channels.py` (the typed-channel processor),
`examples/first_workflow.py` (a `Gated` guardrail and its `Repair`), `src/effective/sql.py` (SQL
as a t-string processor; psycopg takes a `Template` directly on the Postgres side), and
`compose_key` in `src/effective/keys/processor.py`.

## Where documentation lives

| directory | contract |
|---|---|
| `docs/` | maintained deliverables: the first workflow, the concepts tour, the design note. A disagreement with source is a bug |
| `docs/adr/` | the architecture decision records, each the definitive statement of one decision, rewritten in place when the decision moves. A document cites one by number (`ADR-0020 §3`), never by path |
| `wiki/` | what is true now, one concept a page, rewritten in place and interlinked. `wiki/index.md` is the catalog, and its `## Decisions` table is the one ADR index |

What changed, and why, goes in the commit message. Two gates guard citations: `just docs-check`
grades repo-rooted paths and ADR numbers, and `just wiki-lint` grades `[[links]]` and orphan pages.

**Start at `wiki/index.md`**, then `docs/effective-101.md` for the concepts and
`wiki/concepts/architecture.md` for where each subsystem lives.

## Invariants: do not break these

- **Determinism boundary.** Workflows `yield from` typed wrappers (`effective.api`), never a bare
  `yield`. No I/O, clock or randomness between yields: all of it goes through a yielded op.
  `just lint` enforces this.
- **No first-class continuations (`call/cc`).** Durable suspend and resume is done by *replay*:
  re-run the deterministic computation, re-serving recorded op results (the Temporal and Absurd
  model), never by capturing and resuming a continuation. A generator is a *delimited, one-shot,
  explicitly marked* suspension, the tame sliver of continuations with none of `call/cc`'s
  arbitrary-capture and multi-shot hazards. So a layer-injected suspend (the permission
  `AwaitEvent`) resolves on replay, threading state in from *recorded* state, and never by
  resuming a captured layer stack.
- **Three bookkeepers, none derived from another.** Absurd checkpoints are disposable execution
  state. The `ledger` is the canonical append-only record: a trigger blocks UPDATE and DELETE, and
  appends are idempotent by `event_id`. Telemetry is the third and takes the opposite discipline:
  spans are high-volume, disposable and denormalized on purpose, live outside the engine (a JSONL
  sidecar, a collector, or nothing) and are off by default, so no spans table exists. Durability
  governs the first two; no-derivation governs all three, and what telemetry owes in exchange is
  the join column, the placed key. `wiki/concepts/tapes.md`.
- **Absurd facts.** Pull-only: no push, webhook or cron trigger; something external `spawn`s
  tasks. The SDK and `absurd.sql` are **co-versioned**: vendored and pinned at
  `infra/absurd/PIN.txt` (tag 0.5.0, matched to `absurd-sdk` 0.5.0). Checkpoints are JSON, so step
  results round-trip through Pydantic (`to_jsonable_python` and `TypeAdapter`).
- **Keep the core DB-free.** SQLModel and SQLAlchemy live only in `effective.ledger`, never in
  the recording and replay core (`effective` ops, api, and the handlers' base, recording and
  replay).
- **Dependencies run one way.** What yields an op is substrate and lives in `effective`; what
  answers one is an interpreter in `effective.interpreters`, its vendor behind an extra and
  imported only by its own module. `agent`, `examples` and `tui` import the substrate, never the
  reverse. `just lint` checks the direction (`effective.lint --deps`, rule `seam-dep-direction`).
- **Projections are derived.** Workflows do not mutate state: they append events, and projections
  are rebuilt from the ledger. A view over a projection is inspect-only; an authoritative change
  is a new ledger event.
- **Green at each commit**, so history bisects. Never commit secrets (`.envrc`, `.env`).
- **ty-clean.** `just typecheck` runs the pinned `ty`. Sharpen the contract rather than suppress; a
  `# ty: ignore[rule]` carries a documented reason. Use `sqlmodel.col()` for type-checker-clean
  ORM queries.
- **Third-party toolchains: trust, but verify.** A tool from a package ecosystem comes in pinned
  and verified, never floating; npm especially, whose deep dependency graphs make it a richer
  supply-chain target than PyPI. The pattern: an exact version and lockfile integrity
  (`npm ci --ignore-scripts`, never a global install); fetched artifacts hash-recorded in an
  `infra/<tool>/PIN.txt` and corroborated from two independent machines before being trusted; and,
  when the dependency graph is deep enough to hide in, sandboxed execution (rootless Podman,
  `--network=none`, read-only mounts, fetch-and-verify at image build so a runnable image implies a
  verified artifact). Precedents, weakest to fullest: `infra/absurd` (vendored and co-versioned),
  `infra/pg-test` (a digest-pinned image), `infra/formal` (lockfile, cross-machine jar hash,
  hash-gated image build, offline execution).

## Code prose: minimal, and let the types speak

A docstring says what the thing IS plus the one fact the signature cannot. Restating the
signature is duplication that rots, so where a reader could trip on an asymmetry, **rename the
parameter instead of explaining it**.

- **Classic style: show, don't tell.** Punctuation names the relation between clauses: a colon
  says "this explains that", a semicolon "these are coordinate", a period "these are separate".
  **No em-dashes**, which name no relation and leave the reader to infer it. No antithesis of
  your own ("not X, but Y"): a botched one asserts a contrast the sentence never earns.
- **An enumeration is a table**, aligned so it reads in the source, with cases stated positively,
  one row each. Docstrings are MyST markdown for `sphinx-autodoc2`, so the same table renders.
- **Name the thing.** A reference carries its noun ("the second finding", never a bare "the
  second"), and a figure gives way to its fact. A term with one meaning here (`pin`, `totality`)
  stays. US English throughout.
- **No walls of text in code you touch**, comments included. Cut while going over the change.
- **A docstring describes the state of the world.** "Formerly", "no longer", "this was X": each
  states the present badly. Say what holds now and why.
- **No changelog in source.** A dated ruling, a "used to be", a retraction: all of it belongs in
  the commit message. One carve-out: a FIGURE that drifts carries its date and the command that
  recomputes it.
- **Core code states its invariant; it does not cite the tool that enforces it.** "Every site
  minting this variant declares the same role" outlives any gate; a tool documents its own
  enforcement in its own code.
- **Code, tests and formal models cite no ADR.** An ADR's sections move when it is rewritten, so
  code states the invariant in its own words; the wiki carries the connection.
- **Commits are `type(scope): subject`**, lowercase, with a body under about eight non-blank lines.
  Types: `feat`, `fix`, `refactor`, `docs`, `test`. The scope is the subsystem the diff centers on,
  omitted when it spans many. Let the subject carry the result; the diff carries the rest.
- Prose that is genuinely interesting goes to `docs/` or the wiki, where a reader can find it and
  a gate can check its citations.

## Run

`just` lists the recipes.

| recipe | does |
|---|---|
| `just install` | `uv sync` |
| `just check` | the gate: preflight, lint (ruff, the `effective.lint` rules, the f-string gate, `ty`), `docs-check`, `wiki-lint`, `key-check`, the full suite |
| `just test` | the full suite, seed-pinned; Postgres-backed tests skip without a database |
| `just test-fast` | infra-free tests only |
| `just fmt` | format and autofix |
| `just pgt-up` / `just pgt-down` | start and bootstrap, or remove, the digest-pinned test Postgres under rootless Podman (role, database, vendored Absurd schema, migrations) |
| `just pgt-test` | the durable lane with Postgres required: crash at every op, suspend and resume, conformance |
| `just cov` | the 90% coverage gate, against the test Postgres |
| `just formal-setup` | once per machine: quint by `npm ci`, Apalache fetched and sha256-verified (needs node, npm and a JDK 17+) |
| `just formal` | Lean build and quint typecheck, in seconds |
| `just formal-verify` | every registered model check, sandboxed under rootless Podman with no network (`formal-image` builds the image once; `formal-verify-host` runs without Podman) |
| `just migrate` | Alembic migrations against `DATABASE_URL` |
| `just tui-demo`, `just dashboard` | the terminal run view and the run dashboard, on seeded demo runs |

`DATABASE_URL` defaults to `postgresql://effective:effective@localhost:5432/effective`; the test
Postgres may come up on another port (`PGTEST_PORT`), which the recipes read.

## A check licenses only what it covered

Every gate here is cheap to run and cheap to misread, and the misreadings share one shape: the
check's domain was narrower than the claim made from it.

| the move | the discipline |
|---|---|
| running a gate | its verdict is the **exit code**. `just wiki-lint \| grep dead` can print a clean line while the recipe exits 1, because the line the filter dropped is the one that failed it. `cmd >/dev/null 2>&1; echo $?` |
| reading a long log | read the tail, or grep for what you did NOT expect |
| a scripted edit | `assert old in text` before every `str.replace`: a non-matching anchor changes nothing and returns normally. Verify by grepping the text you **inserted** |
| chaining after a filter | `grep -c x file && just check` never runs the gate when `grep` finds nothing, and a gate that did not run looks exactly like a gate that passed. Separate with `;` and echo `$?` for each |

## Environment notes

- **Python 3.14 via `uv`** (`uv run ...`), pinned exactly at `.python-version`. `pyproject.toml`
  sets `exclude-newer = "7 days ago"`, so a dependency resolves only to releases at least a week
  old.
- **`tdom` is vendored** at `infra/tdom`: the sdist-verified 0.1.17 plus one patch, wired through
  `[tool.uv.sources]`. `infra/tdom/PIN.txt` is the record, and carries the command that checks
  `pristine sdist + patch == infra/tdom`.
