# ADR-0025: Race and quorum

- **Date:** 2026-09-18
- **Status:** Accepted. `race`, `quorum` and the deadline (§10) are built on the recorder, replay
  and both engines (`effective.api.race`, `effective.api.quorum`, `effective.choice`); a race
  branch may not yet park (§5).
- **Relates to:** ADR-0008 (gather as structure, whose barrier a race keeps), ADR-0018 (the meter a
  loser's spend reaches), ADR-0020 (key composition, which the race's frame and records join),
  ADR-0024 (the walk).
- **Model:** `formal/quint/race.qnt`.

Each ruling carries what would refute it and, where the model covers it, the invariant of
`formal/quint/race.qnt` that checks it; a ruling the model does not reach says so.

## 1. The decision

`race(branches)` and `quorum(k, branches)` run workflow branches concurrently, record which `k`
succeeded first, and stop the rest cooperatively. `race` is `quorum(1, …)`. They are in-workflow
primitives over work the workflow does. The answer is one of `Chosen`, `TimedOut` (§10) or
`Impossible`, each carrying every branch's ending. An arrival from outside the workflow is a
separate design, the arrivals bridge; the reserved `monitor` arm tag names evaluation over that
evidence and has no op yet.

## 2. What wins, and how each branch ends

| a branch ends in | effect on the race | its ending |
|---|---|---|
| success | eligible to win | `Won` or `Unchosen`, with its value |
| a refusal | a loss; the branch drops out | `Refusal`, with its reason |
| an error a retry could clear | the attempt fails, and the task retries by replay | none: the retry decides |
| an `Unretryable` error, before the choice is saved | the race fails, naming the branch | none |
| an `Unretryable` error, after the choice is saved | none; the choice stands | `Raised` |
| a stop at admission (§5) | none | `Stopped` |

With `s` successes and `u` branches unresolved, the quorum is impossible once `s + u < k`, and the
race answers at once, carrying the successes, the refusals and the unresolved branches.

A value ending carries a digest of the inputs its branch was handed, taken as each input arrives,
so a retry that reaches the same ending must be handed the same inputs. A retry whose branch is
handed different inputs, or inputs with no stable encoding, fails with `EndingLost`.

What counts as the same input is what durability records: an input is its checkpoint encoding,
canonical for what both engines keep. Absurd holds a checkpoint as `jsonb`, which sorts an
object's keys, drops a zero's sign and reads `1e+16` back as an integer, and one attempt is handed
the live value where the next is handed the store's. So an object's key order, a tuple against a
list, a mapping's subclass, a datetime's `fold`, a model's private or excluded field and a signed
zero are all one input, and every NaN is one value. A witness finer than the store fails a retry
that was handed the same input, which is the defect this rules out. A caught refusal is no stored
value, so it is its class, its arguments and its attributes; a part with no encoding is witnessed
by its structure, and one with neither leaves the ending unverifiable. A set is sorted before it
is encoded, since its encoding follows an order the process's hash seed decides and the attempt
that retries is another process.

**Falsified if:** a race answers impossible while `s + u >= k`; an error a retry could clear becomes
an ending; a retry answers with a value its branch derived from inputs whose encodings differ from
the first attempt's; or a retry handed what the store holds of its first attempt's inputs fails.
**Checked by** `impossibleOnlyWhenHopeless`. The input digests are not modeled.

## 3. `want=k`, and the batch

| `k` | behavior |
|---|---|
| `0` | an empty selection, answered without starting any branch |
| `1` to `n` | exactly `k` winners |
| below `0`, or above `n` | refused when the race is built, as a composition error |

The parent reads completions when it wakes. A **batch** is every branch whose result is available
at that wake, read in one pass and ranked by branch index. Which branches land in a batch depends
on the schedule. The batch is recorded beside the choice to explain it, and is never served:
before the choice is saved, a re-run reads completions afresh. Winners are returned in branch-index
order.

**Falsified if:** a race returns other than `k` winners without answering impossible; returns
winners out of branch-index order; or a replay is served a batch.
**Checked by** `exactlyK` and `zeroStartsNothing`. The batch is not modeled.

## 4. The choice

1. The parent saves the choice, one checkpoint naming the winners or the impossibility, together
   with its **horizon** (§5). The race is decided at that commit.
2. Then it publishes the choice to every branch.
3. What the store returned is the choice. Anything that acts on it reads that value, never what it
   attempted to write.

A crash before the save re-runs the race, and a different choice is then legitimate. Every call a
branch finished is served from its checkpoints, so a re-run favors the branches that had made
progress. A crash after the save serves the choice, and a loser stops only for a choice the store
holds.

**Falsified if:** a replay after the save records a different choice; a crash before it leaves a
choice in the record; or a loser stops for a choice the store does not hold.
**Checked by** `choiceStable`, `everyStopHasItsChoice` and `noWinnerWasStopped`.

## 5. Cooperative cancellation

A loser is checked twice, and each check names what it reads by its own count:

| check | where | lets through | counted by |
|---|---|---|---|
| walk | each op the workflow yields, before the layers | an op the loser's horizon covers | the handler's walk |
| base | where an op reaches the engine, when the store holds no checkpoint for it | an effect, only while no stored choice names the thread a loser | the engine |

An op is **started** once it passes the base check, and only a started op's effect can run.

The **horizon** is one count per loser thread of control that was running when the choice was
saved: a race or gather branch's handler, which a scoped body continues. A finished child's count
folds into its parent's, so a replay of it stays bounded.

| a loser, once a stored choice names it | does |
|---|---|
| an op it had started | finishes it, its layers' code after the engine answers included |
| an op the walk check let through, not yet at the base check | stops at the base check; the domain is not called |
| a walk op within its horizon, on a retry | replays it, served from the record |
| anything past its horizon: a leaf, a structure, an op a layer yields after a resume | stops (a step budget of 0) |
| an op a crash interrupted after the choice | never reruns it; what the op committed stays, and it has no result |

A stop is a value at the barrier. It escapes the layer driver without being thrown into a layer,
and every layer generator is closed, so a layer's `finally` runs. Nothing preempts a thread, so a
stop is never an unwound stack, which is the no-continuations invariant.

A race branch may not park yet: an await or a sleep is refused by its kind. When parking is built,
a parked loser sees the choice when it is next driven and stops without waking. A race never
stops its loser mid-op. ADR-0026's in-flight cancel records a distinct `Cancelled` result for an op a face
stops; a loser's started op does not use it, and that extension is deferred.

**Falsified if:** a loser runs an effect after it observes a stored choice naming it a loser, an
op a layer injects included; a stopped loser takes a step past its horizon; an op started by the
incarnation that returns has no recorded result after the barrier; or a crash-interrupted op
reruns.
**Checked by** `noAdmitAfterFlag` and `admittedRecordedAtBarrier`. The model has one check and no
layers, so the two checks, the horizon and the step budget are not modeled.

## 6. When the race returns

At the barrier, once every loser has stopped and no loser op is in flight, as `gather` does. Its
latency is the longest in-flight started op of any loser, and its record is sealed on return.
Returning at the choice while losers drain is deferred.

**Falsified if:** the race returns while a loser has an op in flight or can still start one.
**Checked by** `returnsQuiescent`.

## 7. The race's record

Everything each loser committed: its checkpoints, and its ledger rows by `writer_placement`, a row
committed before its checkpoint included. A loser's spend reaches the run's meter at the barrier,
folded in branch-index order as a gather branch's is. An opaque effect a tool performs is outside
the record.

**Falsified if:** a loser's spend never reaches the root meter, or a loser's ledger row carries no
placement that names it.
**Checked by** `loserSpendReachesMeter`.

## 8. Nesting

| an inner race, when its enclosing loser's choice arrives | does |
|---|---|
| has not saved its choice | never saves one; its branches stop at admission |
| has saved its choice | keeps it on record; its enclosing loser then stops |

Every race in one tree of nested races decides under one lock: it checks whether an enclosing choice
has stopped it, saves its own choice and publishes it while holding that lock, so no inner save
lands after an enclosing loser's choice. On replay, a loser's inner race with no saved choice
starts nothing.

**Falsified if:** an inner race saves a choice after its enclosing loser's choice was published.
**Checked by** `noInnerChoiceAfterOuterFlag`.

## 9. Identity: the records a race writes

A branch frame is `race:{r},{i}`, beside `gather:{g},{i}` in `FRAME_ARMS`, with its own ordinal.
Each record sits inside the frames that enclose it:

| record | named by | written when | holds |
|---|---|---|---|
| `race:{r};choice` | the race | the race decides | the winners or the impossibility, the batch, the horizon |
| `race:{r};endings` | the race | at the barrier, by the first incarnation to reach it | each branch's ending kind, a refusal's reason or an error's text, and a value ending's input digest; never a value |
| `refusal;{resolved}` | the name the engine resolved for the op | the domain refuses a branch's op | the reason, and whether a later attempt serves it (`Refused`) or calls the domain again (a subclass) |
| `gated;{placed}` | the op's walk name | a layer refuses a branch's walk op | the reason |

Checkpoints stay bare values, inside a race and out. A **fresh** walk reads neither refusal
record: one at attempt 1, in a race whose choice the store does not hold. A resume after a park
runs on the attempt that parked, so the attempt alone does not say that no earlier execution wrote
a record. What makes the rule safe is that a race which wrote records and saved no choice fails
its attempt: only refusals are delivered to the workflow, so an error before the choice ends the
attempt whatever the workflow catches, and the next execution is a later attempt.

**Falsified if:** a race frame's key, or any of these records, aliases another race's or a
gather's; a horizon grows with a loser's ops rather than its live threads; a fresh walk reads a
refusal record; or a walk that is not fresh skips one.
Not modeled.

## 10. The deadline

`race(branches, deadline=at)` and `quorum(k, branches, deadline=at)`. A keyword rather than a
wrapper, because the timeout is an outcome of the same choice: a wrapper would say it composes
over any effect, which is what the second rule below refuses. Keyword-only, as `await_until`'s
deadline is, so every bound names its instant.

| the rule | why |
|---|---|
| processing time | event time waits for the evidence algebra |
| the deadline is an absolute instant the caller hands in, and a naive datetime is refused | a restart cannot extend it, below; a naive one names a different instant on a worker in another zone |
| a timeout is an outcome in the same choice checkpoint | one authority decides |
| a completion at exactly the deadline is a timeout | the bound is the instant, not the interval |
| the deadline is read before the first branch runs and at every wake after it | a race whose deadline is already behind it starts nothing, which is the state a crash before the choice leaves a retry in |
| on a ctx whose branches cannot overlap, a branch already running when the deadline arrives finishes | a sequential interpreter reads the deadline between branches and nowhere else, and a deployed Absurd worker takes that path |

**The instant is what a restart cannot extend, and it is already durable when the race sees it.**
A workflow reads its clock through a `step` and hands the answer in, which is `await_until`'s
contract: the step's checkpoint holds the instant, so every attempt races against the one the
first attempt chose, and a reader asking why a race timed out reads that step. A race that saved
the deadline again when it started would keep a second copy of a value the store already holds and
the workflow already reproduces. Two copies of one deadline can disagree, and a pair that drifts
apart expires a wait that was not the one either was written for.

**`deadline`, not `within` or `timeout`.** A deadline is a point in time and a timeout a duration,
which is the distinction a caller has to get right and the one `await_until` already keeps: the
substrate takes an instant everywhere it takes a bound. `within` would also read beside
`combinators.within_budget`, which bounds something else.

**Falsified if:** a race restarted after a crash times out later than the instant it was given.
**Checked by** `deadlineNotExtended`, `winnersBeforeDeadline` and `timeoutOnlyAtDeadline` in the
model, and in the tree by `tests/test_race_deadline.py` and `tests/test_race_durable.py`, which
hold the one clock a race reads rather than waiting for a real one.

## 11. Layers and gates

A race branch's layers are assumed to decide the same way on every attempt, as layers are
everywhere. A changed decision that would let a refused op run fails the task; the others are
bounded as the table says:

| a layer, on an earlier attempt | on this one | outcome |
|---|---|---|
| refused an op | forwards it, rewritten or not | `RefusalDiverged` before any write, for a step, a ledger row and an artifact alike |
| refused an op | answers it itself | `RefusalDiverged` |
| refused an op | injects another op while handling it | `RefusalDiverged` |
| forwarded or answered an op | refuses it, while its loser replays | `RefusalDiverged` |
| answered an op | forwards it | a loser stops at the base check; a winner calls the domain, as outside a race |
| answered an op | answers it differently | the branch's inputs changed, so a value ending fails with `EndingLost` (§2) |
| answered an op | answers it the same way | replayed; no record is kept of a layer's answer |

A domain's refusal is recorded (§9) and served on a later attempt without calling the domain, and
the engine counts the served occurrence as it counted the refused call. A layer that injects an op
while handling one it then refuses needs no declared provenance until the first such layer is
written; until then the rejection above covers it.

**Falsified if:** a retry calls the domain for an op whose refusal is recorded, or a changed gate
decision lets a write through.
Not modeled.

## 12. Synchronization

No flag two threads share is read without a lock, and nothing rests on the atomicity of a
builtin. A handler's own walk state is confined to its thread, and its parent reads it only after
the branch has joined. Each branch's
cursor has its own lock; a race seals its live cursors while it saves its choice and unseals them
even when the save fails. Locks are taken in the order race tree, cursor, engine, and a cursor's
lock is never held while another lock is taken. The concurrency target is free-threaded CPython.

**Falsified if:** a race answers differently, or a loser runs an effect after its choice, on
free-threaded CPython with the GIL off.
**Checked by** `just race-free-threaded`, the race's tests with the GIL off. Not modeled.

## 13. What this does not rule

| left to | what |
|---|---|
| the arrivals bridge | subscriptions, delivery and acknowledgement, and formulas over outside evidence |
| per-branch budget partition | how a budget is shared among a race's branches; a budgeted race waits on it |
| deferred designs | returning before quiescence, stopping mid-stream, racing several parked events, and branch-and-bound and best-first |
