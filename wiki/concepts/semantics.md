# The handler is the semantics

A workflow is a term and a handler is its meaning. The op set is the abstract syntax, the handler's
`match` is a small-step transition relation over it, and replay, crash recovery and park/resume are
that one relation run against a populated store. The full statement, with its rules in judgment
form, is [`docs/effective-design.md`](../../docs/effective-design.md) §3; the decision to write it down and divide its proof
obligations across tools is [ADR-0016](../../docs/adr/0016-formalization-and-operational-semantics.md).

## Syntax and configuration

The syntax is the closed union `WorkflowOp` in [`src/effective/ops.py`](../../src/effective/ops.py): inert frozen dataclasses
that perform no I/O. A workflow reaches them only through the typed wrappers in
[`src/effective/api.py`](../../src/effective/api.py), and `just lint` holds every registered workflow file to that. Layers
yield ops directly, since a layer stands between the workflow and the handler.

A configuration is $\kappa = \langle W, C, L, E \rangle$:

| part | is | discipline |
|---|---|---|
| $W$ | the workflow's remaining computation | re-run from the top on every resume |
| $C : \mathsf{Name} \rightharpoonup \mathsf{Val}$ | the checkpoint store | disposable execution state |
| $L \in \mathsf{Row}^{*}$ | the ledger | canonical and append-only |
| $E : \mathsf{Event} \rightharpoonup \mathsf{Val}$ | delivered events | read by an await |

No rule reads $C$ to produce $L$ or the reverse, which is why a checkpoint may be collected and a
ledger row may not ([concepts/tapes](tapes.md)).

## The rules

Every op is answered from the tape when the tape holds its name, and from the world otherwise:

```math
\frac{n \notin \operatorname{dom}(C) \quad \tau \Downarrow v}
     {\langle \mathsf{step}(n,\tau);W,\ C,\ L,\ E\rangle \rightarrow \langle W[v/\bullet],\ C[n\mapsto v],\ L,\ E\rangle}
\qquad
\frac{C(n) = v}
     {\langle \mathsf{step}(n,\tau);W,\ C,\ L,\ E\rangle \rightarrow \langle W[v/\bullet],\ C,\ L,\ E\rangle}
```

The second rule has no premise on $\tau$, so a recorded effect is never evaluated again. That
premise is the whole of durability.

| rule | fires when | result |
|---|---|---|
| Step-Run | the name has no checkpoint | the effect runs and its value is checkpointed |
| Step-Replay | the name has a checkpoint | the checkpoint is served and nothing runs |
| Await-Resume | the event was delivered | its payload is sent in |
| Await-Park | the event was not delivered | the run parks; the worker may exit |
| Ledger-Append | the row's `event_id` is new | the row is appended |
| Ledger-Idem | the row's `event_id` is present | the ledger is unchanged, and within one task both engines refuse the run; `RecordingHandler` appends the row again |
| Gather | each branch runs under the prefix `gather:g,i;` | results join by branch index, and the branches' checkpoint domains must be disjoint |
| Crash | a worker dies at op $j$ | committed $C$ and $L$ survive; the in-flight frame is discarded |
| Resume | a worker claims the task | $W$ re-runs from the start, and every committed name takes Step-Replay |

A park is a crash the workflow asked for. Resume re-runs the workflow, re-derives the await's name
from recorded state, and finds the payload under it: [`docs/effective-101.md`](../../docs/effective-101.md) §5.3 draws the
sequence. The Gather rule's disjointness side condition is what key injectivity buys
([`docs/effective-design.md`](../../docs/effective-design.md) §5); disjoint names still say nothing about two branches touching one
outside row.

## Handlers refine one relation

| handler | relation | where |
|---|---|---|
| `RecordingHandler` | $\rightarrow_R$, in memory, canned or live answers | [`src/effective/handlers/recording.py`](../../src/effective/handlers/recording.py) |
| `ReplayHandler` | $\rightarrow_P$, answers only from a trace, raises `ReplayMismatch` on a moved key | [`src/effective/handlers/replay.py`](../../src/effective/handlers/replay.py) |
| `DurableHandler` on Absurd and Postgres | $\rightarrow_A$ | [`src/effective/handlers/durable.py`](../../src/effective/handlers/durable.py), [`src/effective/engines/absurd.py`](../../src/effective/engines/absurd.py) |
| `DurableHandler` on embedded SQLite | $\rightarrow_S$ | [`src/effective/engines/sqlite.py`](../../src/effective/engines/sqlite.py) |

For a well-formed workflow, one whose rows have distinct event ids, the four are meant to agree on
the committed ledger up to a quotient, $L(\Downarrow_R W) \approx L(\Downarrow_P W)
\approx L(\Downarrow_A W) \approx L(\Downarrow_S W)$, with $\Downarrow$ read as driven to completion,
wakes included. The relation is $\approx$ because a concurrent gather commits its branches' rows in
a schedule the handler does not choose: the observable is a labelled partial order, and a recorded
sequence is one linear extension of it. [`docs/effective-design.md`](../../docs/effective-design.md) §3.5 states the quotient, the
reader obligation it puts on every fold, and the cases where a handler realizes strictly less. Two
of those cases matter here: `ReplayHandler` keeps no ledger, and a repeated event id completes with
two rows under `RecordingHandler` while both engines refuse it. The agreement is tested for the
rows' multiset; their precedence is intended and untested.

## Two refusals that follow

**No first-class continuations.** Suspension happens at a marked `yield`, resumes once, and spans a
delimited extent, so a durable engine stores names and values and never a frame. A layer-injected
park resolves on replay like any other ([concepts/machine](machine.md)).

**Color lives in the handler.** The workflow is a plain generator with no `async`. A handler calls
its interpreters synchronously, and an asyncio interpreter hands its coroutine to a loop on its own
thread ([`src/effective/interpreters/aio.py`](../../src/effective/interpreters/aio.py)), so the workflow and the handler never see the loop.
A `gather` inside a workflow therefore needs no `async def`.

## The laws, and how each is known

[concepts/claims](claims.md) carries each law's scope and assumptions; [concepts/testing](testing.md) carries the
instruments.

| law | rule it protects | evidence |
|---|---|---|
| composed names do not alias | Gather's side condition | proved ([`Keys.lean`](../../formal/lean/Effective/Keys.lean)); the string serializer sampled |
| a step is deterministic for a fixed oracle, and a checkpoint forces replay | Step-Run, Step-Replay | proved ([`Step.lean`](../../formal/lean/Effective/Step.lean): `step_deterministic`, `replay_stable`) |
| the ledger only appends, idempotent by `event_id`, and neither bookkeeper is a function of the other | Ledger-Append, Ledger-Idem | proved ([`Ledger.lean`](../../formal/lean/Effective/Ledger.lean)); the trigger tested on both engines |
| a gather terminates under fairness and never deadlocks | Gather | model-checked ([`gather.qnt`](../../formal/quint/gather.qnt)) |
| a revived worker awaits the name its recorded state implies | Await-Park, Resume | model-checked ([`govern_park.qnt`](../../formal/quint/govern_park.qnt), `awaitingMatchesPass`) |
| accrual across branches is schedule-independent | Gather, with a meter | proved ([`Budget.lean`](../../formal/lean/Effective/Budget.lean)) |
| a recorded result is never re-executed | Crash, Resume | tested by a crash at every op on both engines ([`tests/test_replay_crash.py`](../../tests/test_replay_crash.py), [`tests/test_shape_conformance.py`](../../tests/test_shape_conformance.py)) |
| the handlers agree up to the quotient | the refinement above | tested by one conformance suite on both engines ([`tests/_conformance.py`](../../tests/_conformance.py)); the multiset half is pinned for the ledger, the precedence half for checkpoints only |
| a gather is deterministic, joined by branch index | Gather | open: no Lean theorem, and no composition lemma connects `Step.lean` to `gather.qnt` |
