# Three tapes, one address

An evolving run is described by three append-only records, and they are joined by **one computable
address** rather than by correlation. Projection is the operation that connects them;
[concepts/graph](graph.md) is the algebra it uses once the rows are in hand.

| tape | holds | discipline | store |
|---|---|---|---|
| **checkpoints** | how the computation went: the placed key of every op that reached `ctx.step` | disposable execution state, thrown away freely; it re-serves a replay and settles nothing | the engine's own table |
| **ledger** | what was decided: the canonical semantic record | append-only, a trigger blocks UPDATE/DELETE, appends idempotent by `event_id` | `ledger`, in the engine's database |
| **telemetry** | what it cost and how it behaved | high-volume, disposable, denormalized on purpose, off by default | **outside the engine.** A JSONL sidecar, a collector, or nothing. No spans table exists |

A checkpoint holds whatever the op answered, and an op stopped while it ran answers `Cancelled`, so the tape can hold a cancel that a replay serves without running the op again.

**The rules split along durability, and the split is deliberate.** No bookkeeper is derived from
another, and that governs all three. Durability governs the first two only: the durable-record
discipline does not govern the span stream, and telemetry's side of the bargain is the join column. A truncated final span line is an ordinary outcome, because the process
writing it is one this repo kills at every op. The ledger would never tolerate that.

`AGENTS.md` names two durable bookkeepers and a third that is not; the third is also named in
source (`dashboard.py`).

## Why they can be joined at all

The tapes carry **the same address**, not two addresses that have to be matched up. A span records
the placed key of the op it observes, which is what the checkpoint tape is made of and what a
`ledger` row's `event_id` is composed by. Two computable addresses do not need to be correlated;
they need to be the same address.

That is the identity grammar paying off **across** stores rather than inside one, and it is why the
join survives a storage decision nobody has made yet: spans can move to Postgres, a sidecar or a
collector without changing anything above, since the join is a key either way.

## What is built

| pair | state |
|---|---|
| checkpoints ⋈ spans | **built**, conformance-pinned on both engines, and it folds correctly through `fold_cycles`. A left outer join **from the tape**: a node with no span is ordinary, a span addressing no node is a defect and asserts empty |
| ledger ⋈ spans | **not built**, and the two are governed by opposite disciplines |
| ledger ⋈ checkpoints | **forbidden as a derivation.** `graphview.ledger_collisions` is a *detector of disagreement* read off the tape alone, which is the shape the invariant leaves available |
| one projection over all three | **not built.** The AR firm joins two of them, ledger and checkpoints, with the run's spawn params (`src/examples/agent_firm/factors.py`); assembling the graph object is open |

The two read paths are disjoint in code: a domain's projection folds the ledger, and `graphview`
folds checkpoints with spans.

**The built join has a hole where the system ships.** A strict `xfail` fires when an op layer
short-circuits, because `_place` counts every op routed through the layer stack while the engines
count only ops reaching `ctx.step`. The shipped permission cascade is such a layer, and unifying the two placement minters is open.

## What a tape weighs

A tape's weight is what its tools return. Keys stay a small fraction of it, and a tool that
returns more (twenty search results with snippets, a page's links) grows the tape in proportion,
whether or not the workflow reads all of it. Against the work it stays small: it holds the answers
to every fetch and judgment the run paid for, and is usually smaller than the spans beside it and
much smaller than the raw responses those tools received. Losing a tape costs a re-record; losing
answers costs the work, which is what a content cache keeps.

## Where the order obligations land

This is why the three tapes are one subject rather than three.

The reader obligation is [`docs/effective-design.md`](../../docs/effective-design.md) §3.5, and which folds pay it is a test rather
than an argument ([`tests/test_reader_quotient.py`](../../tests/test_reader_quotient.py)). What belongs here is that it is not special to the ledger:
it falls on **every tape a projection folds**, and the three differ in how much order they have to
give:

| tape | order it carries |
|---|---|
| ledger | a partial order, with the recorded sequence one linear extension of it. `seq` is the stored order and **not** commit order |
| checkpoints | the same quotient under a gather, asserted over `checkpoint_keys` rather than over rows |
| telemetry | **none.** `span_id` is a hash rather than an ordinal and collides by construction; emission order is the only order there is |

So: **the join is safe and the folds are not.** Joining on an address is order-free, which is why
the address was the right mechanism; folding is where the law bites, and last-write-wins over a
shared key, the default projection idiom, fails it by definition.

The live exposure is **cross-task rather than gather**: a projection reading every run's rows folds
events from independent tasks, which are incomparable by construction and never claimed an order. A
revision is the exception, spawned only after the decision it revises committed, so those two are
causally ordered, and a last-write-wins fold over revisions is sound by exactly that order.

## The telemetry format

A span file is OTLP/JSON: each line is one complete `TracesData` object, the format the OTLP file
exporter writes, so a run needs no collector to leave a readable trace. `otlp_jsonl_sink` appends
the lines and `check_otlp_line` says how a line fails to be one
([`src/effective/telemetry.py`](../../src/effective/telemetry.py)).

| attribute | names |
|---|---|
| `gen_ai.*` | what the OpenTelemetry GenAI semantic conventions define: the operation, the conversation, the agent, the tool, the messages ([references](../references.md)) |
| `effective.*` | what is Effective's own, such as the iteration and the cost of a reused answer |
| `effective.key` | the placed key, the join column to the other two tapes, flattened once at the sink |

A span whose op the engine placed carries `effective.key`; the recording/replay core places
nothing, so its spans carry none. The core imports no vendor SDK. The readers this format targets
are backends that ingest OTLP, such as LangSmith, Pydantic Logfire and MLflow tracing; an exporter
that sends spans to a collector is not built.

## See also

[concepts/graph](graph.md) for the projection algebra, [concepts/machine](machine.md) for why a replay re-serves the
checkpoint tape rather than resuming a frame, [concepts/evidence](evidence.md) for why a built join still owes
an anti-vacuity count. [`docs/effective-design.md`](../../docs/effective-design.md) §3.3 and §3.5 carry the partial-order semantics.
