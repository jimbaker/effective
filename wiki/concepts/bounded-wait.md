# What decides a bounded wait

`await_until` has two endings where `await_event` has one, and every arm of that choice has been
wrong at least once. Four review rounds each found a defect of the same shape: a state the two
engines answered differently, in a cell nobody had written down. This page writes them down.

**Absurd is the reference and SQLite conforms**, so every row below is one answer, not two, and
the cross-engine pin is what says so. `ops.WaitOutcome` is the union; `SqliteTaskContext.await_until`
and `engines.absurd.sdk_await_until` are the two implementations.

## The decision, in order

The first row that matches answers, so the order is the rule rather than a presentation of it.

Rows four and five reach the same ANSWER by different routes, and row four is pinned on the
absence of a park because nothing else tells them apart: disable the deadline read and the wait
parks on an instant already gone, is claimable at once, and row two answers `Expired` on the next
claim. Same answer, one extra claim, and every cross-engine history stays green.

**That pin is the embedded engine's, and the reference takes the extra claim.** `absurd.await_event`
has no deadline arm at all: a first call with no stored event registers a wait and suspends
whatever the clock says, and row two answers it next time round. So the ANSWER is one and the
number of claims is two, which is the one place this page's ordered table describes the engines
and not just the rule. No cross-engine history pins the reference's extra claim yet.

| the wait | answers | pinned by |
|---|---|---|
| has settled | what it settled, whatever has happened since | `test_a_bounded_wait_that_expired_ignores_a_later_event_on_both_engines` |
| is open, and the clock woke this claim | `Expired` | `test_an_event_after_the_deadline_leaves_an_expired_wait_alone_on_both_engines` |
| is open, and its event is stored | `Arrived(payload)` | `test_a_bounded_wait_takes_an_event_already_emitted_on_both_engines` |
| is open, no event, and its deadline has passed | `Expired`. The embedded engine settles it here, the reference on the claim after | `test_a_deadline_already_past_expires_without_parking`, on the embedded engine |
| is open, no event, and its deadline is ahead | parks on both | `test_every_history_in_the_bounded_wait_table_answers_alike_on_both_engines` |

Row one is what makes a wait replayable: the outcome is recorded when it is first reached, so a
replay serves it rather than deciding again. **What enforces it is the settle.** Both engines
write insert-or-ignore and return what the store holds, so a second decision reaches the first
answer whether or not the read above it happened; deleting that read leaves every pin green.
Worth knowing before someone reaches for the read as the guarantee.

Row two before row three is the reference's order and it is the one that keeps costing. The SDK
raises its timeout from `wake_event` **before** it reads the events table, so a task its deadline
woke expires even where the event has since landed. Put row three first and a late event answers a
wait the clock already ended.

Row three before row four is the same order read the other way: a wait on its FIRST call has no
wake to consult, so the store answers it whatever the clock says.

## The wake, whose lifetime is where the defects lived

Row two asks *did the clock wake this claim*. Absurd keeps that as `wake_event` on the run; SQLite
keeps it as `tasks.waiting_event` and re-derives the answer, so the column's lifetime **is** the
invariant.

| the event | the wake | why | pinned by |
|---|---|---|---|
| a park registers on a name | written | the park is what the wake belongs to | `test_a_park_records_its_deadline` |
| an emit lands, deadline ahead | cleared, and the event answers | the waiter is owed its wake | `test_an_event_before_the_deadline_arrives_with_its_payload` |
| an emit lands, deadline passed | left standing | `absurd.emit_event` deletes those waits and wakes the rest | `test_an_emit_past_a_deadline_leaves_the_waiter_where_it_was` |
| a claim takes a `waiting` row | kept | the deadline disjunct is the only way one is claimed, so the clock got here | `test_an_event_a_wait_never_parked_for_is_taken_on_both_engines` |
| a claim takes anything else | cleared | a wake belongs to the attempt it woke, and this claim is a different one. `absurd.fail_run` gives the next run `wake_event` NULL, and `claim_task` never re-claims a `running` one | `test_a_wake_does_not_outlive_the_park_it_ended_on_both_engines`; the RECLAIM half is SQLite-only (`test_a_reclaim_reads_the_store_rather_than_a_dead_claims_wake`), since no history drives a worker death |
| the wait it answers reads it | spent for THIS claim | one wake answers one ask, and the record of spending it need not outlive the claim the column does not outlive either | `test_a_wake_is_spent_by_the_wait_it_answers` |
| the SDK's expiry fires | spent on the RUN, after the outcome settles | the SDK clears only the dict it holds, so the column goes on naming an event this run was already woken for, and `absurd.await_event`'s resumed-due-to-timeout arm answers a later ask with a null payload | `test_every_history_in_the_bounded_wait_table_answers_alike_on_both_engines` |

`_claim_locked`'s `CASE` is the one expression that has to know this, and the claim is the right
home for it because a claim is the event that decides: a list of park-ending writers is bounded by
what it enumerates, and the lease reclaim was outside it.

**A reader of the column asks a state as well**, since a park is a state and the column
outlives one. The three readers ask two different questions and the difference is deliberate.

| the reader | its question | what it asks |
|---|---|---|
| `parked.read_sqlite_parked` | what can an emit still answer? | `waiting`, a name, and a deadline still ahead |
| `bridge_sqlite.park_name` | the same, of one task | it PROJECTS that relation rather than restating it |
| `runs.read_sqlite_runs` | what is this run parked on? | `waiting` and a name. A run whose deadline is running out is parked on that event right up to the claim that ends it |

Two of them once read the column raw and reported a sleeping task as parked on an event nobody
was waiting for; `park_name` then asked the state alone and named parks the first relation
excludes.

## What the deadline's precision actually is

**Every instant the reference stores is a microsecond `timestamptz`.** `w_{queue}.timeout_at`,
`r_{queue}.available_at` and `e_{queue}.emitted_at` all report `datetime_precision = 6`. The
whole second is one parameter: `absurd.await_event` takes `p_timeout integer`, relative to the
claim, so the park it registers ends at `claim + ceil(deadline - claim)`.

Two readers take that instant for the deadline, and each turns the rounding into an answer.

| the reader | what the instant decides |
|---|---|
| `absurd.emit_event` | which waits it deletes as late, and which it WAKES |
| `absurd.claim_task` | when the run becomes claimable again |

So `engines.absurd.sdk_pin_the_park` writes the deadline over both columns as an absolute
`timestamptz`, on the `SuspendTask` the park raises. A correction that is merely LATE is not
harmless, which is why it is not allowed to be late: see the transaction below.

What is left is the gap between the two clocks that compare: `time.time()` on the embedded
engine, `absurd.current_time()` on the reference. They share a host here, and the gap measured
0.001 ms median and 0.325 ms worst over 40 reads, round trip included (2026-09-20). On a managed
Postgres they do not share one and it widens by the skew. Recompute it with:

```
uv run python -c "import time,psycopg;c=psycopg.connect('postgresql://effective:effective@localhost:5432/effective',autocommit=True);b=time.time();p=float(c.execute('select extract(epoch from absurd.current_time())').fetchone()[0]);print(f'{(p-(b+time.time())/2)*1000:.3f} ms')"
```

A cross-engine history that turns on the clock waits `PAST_THE_SKEW` (0.2 s), which is three
orders above that gap and small enough to fail the timer this replaced: at that margin the
reference answered `arrived` three times of three before the correction landed. It is an
empirical allowance on a shared host, not a bound anyone has demonstrated for a loaded CI runner
or for a managed Postgres.

**A rule about the instant itself does not take the allowance**, because both instants can be the
engine's. Emit under a held clock and the tie is exact: an event a microsecond before the deadline
arrives, one AT the deadline and one after expire, which is the race's tie rule (a timeout and a winner decided by one authority, at one instant) read
at the resolution the engine stores. [`tests/test_absurd_deadline_park.py`](../../tests/test_absurd_deadline_park.py).

Punctuality is a floor of its own and a different one. The instant is now the one the workflow
named; how soon a worker NOTICES it is the driver's business, and nothing here establishes a
bound. The SDK's own worker loop polls every 0.25 s by default and this repo does not run it:
tests drive `work_batch` by hand, and a deployment drives its own worker loop. So the page states the
instant and says nothing about the latency.

**The park and the deadline it ends at are ONE transaction**, because the SDK connects with
autocommit and two commits leave the wait live on the rounded instant for as long as the worker
takes to reach the second one. That is progress, not a clock tick, so a pause or a death inside it
is enough, and everything that reads a wait acts on what it finds: `absurd.emit_event` takes such a
wait, writes the SDK's arrival checkpoint and deletes the row, so no later claim asks again, and a
claim on the same run can settle that wait and park another one the stale write would reach.
`run_id` names a RUN, which outlives the claim that parked it. Both reproduce, and both are
answered by the one transaction;
[`tests/test_absurd_deadline_park.py`](../../tests/test_absurd_deadline_park.py) pins it with an emit racing the park on its own connection.

**An absolute park is also what makes a deadline testable.** `absurd.fake_now` is the hook the
engine ships for holding its clock, and `tests/_durable.clock_at` holds both clocks a park
consults. A park registered RELATIVE to the claim lands at `fake_now + ceil(...)`, which carries
however far the fake clock was moved: measured at 3600.000000 s past the deadline for a clock held
an hour on, with the run still `sleeping` after the clock was advanced TO the deadline. Registered
absolutely it lands 0.000000 s off and that same advance expires it. So a rule about the instant
itself costs no wall-clock time to pin, which is what the race's tie rule needs.
[`tests/test_absurd_deadline_park.py`](../../tests/test_absurd_deadline_park.py).

Found by the table, not by a case: every per-history pin happened to use numbers outside the
window, which is what an enumeration is for.

## The slot, which is a different axis

*Which* wait is being decided is not *what* it decides. A wait settles under the op's placement
(`DurableHandler._wait_slot`), never under the event's address: two asks of one name are one
address and two questions, which is the shape a receive loop has. A layer may inject an
`AwaitEvent` of its own, and there the ambient placement belongs to the op the layer wrapped.

| the wait | settles under | pinned by |
|---|---|---|
| the first ask of a name | `event;{name}` | `test_an_authored_wait_settles_under_its_own_name` |
| a later ask of the same name | `event;{name}#k` | `test_two_bounded_waits_on_one_name_answer_independently_on_both_engines` |
| one a layer injected | its own placement, never the wrapped op's | `test_a_layer_injected_wait_settles_clear_of_the_op_it_wraps` |

## Where a bounded wait is refused

Three refuse it FOR ITS DEADLINE, each because the wait's two endings need something the context
cannot give. A fourth refusal reaches it first and is about the name rather than the clock.

| the context | the refusal |
|---|---|
| a gather branch | a branch resolves its wait by peeking, and a peek reads the event alone |
| a race branch | a race branch may not park at all, refused by the op's kind |
| a counterfactual's tail | a grant says what arrived, a deadline asks whether it arrived in time, and a fork spends no time (`ForkedDeadline`) |
| an ABSOLUTE name under a fork child's rename | `refuse_absolute_await_under_a_rename`, which fires ahead of the deadline arm: the name would be prepended with an event world its emitter has never seen, so there is no wait to bound |
