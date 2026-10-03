# What a forced schedule can decide

`tests/_schedules.py` turns the op-layer seam into an instrument: a `Turnstile` blocks before its
`yield op` and releases ops in a named order, so a test asserts over two named interleavings rather
than over whichever one the run took. It orders **admissions**. That is its domain, and the
question a row has to answer first is whether the thing it wants to decide is an admission.

| the shape                     | the order decides it | because                                       |
|-------------------------------|----------------------|-----------------------------------------------|
| a gather's commit order        | yes                  | each branch's ops are admissions               |
| a last-write-wins reader       | yes                  | it reads what the commits left                 |
| a meter's fold                 | yes                  | a branch's spend folds at the barrier it reached |
| a race's winner                | **no**               | a race turns on COMPLETIONS, below             |
| anything behind a blocking domain call | **no**       | a turn spans the call, so the branch holds it  |

A schedule is not the only instrument, and the second one reaches what it cannot.

| the shape                     | a held CLOCK decides it | because                                     |
|-------------------------------|-------------------------|---------------------------------------------|
| a race's deadline firing       | yes                     | `handlers.base.race_clock` is the only clock a race reads |
| which instant a branch ended on | yes                    | the branch moves the held clock as it ends  |
| a race's winner, still         | **no**                  | the completions the clock stamps are not the completions it orders |

## Why a race is outside it

Three facts, each measured on SQLite while putting `race`/`quorum` through the shape table.

| the fact | what it costs a schedule |
|---|---|
| the turnstile releases the next turn from its layer's `finally`, which runs BEFORE the branch body returns | ordering a branch's last step first does not put its completion in an earlier batch |
| `decide` reads a BATCH, and ranks the branches in one batch by index | a schedule that puts branch 1 first still answers branch 0 whenever both land together: **5 of 200** runs under CPU contention, against 200 of 200 for the order the index already agrees with |
| a race that lets every branch finish stops nobody, so the loser ends `unchosen` | a schedule cannot even guarantee that cancellation HAPPENS, which is the thing the row exists to exercise |

A turn also spans the domain call, so the obvious repair fails too: hold the loser in its domain
until the choice lands and it holds its turn, and the order deadlocks.

## What does force a race

The two-event hold, which `tests/test_race_durable.py` established and `tests/test_race_shapes.py`
reuses. The winner's first call waits until the loser is inside its own first call; the loser's
call waits for the choice. Then the loser's next op is admitted, or not, after the flag, and which
it is has no timer in it.

Key the release on the **choice** rather than on any settle. `_Settled` fires on every checkpoint,
so a loser is released by whatever the winner happened to write first; `_ReleasedByChoice` names
the event the assertion is about.

**What the hold measures that the schedule could not.** Two runs of one race, each with the
other branch held inside its first call until the choice: the answer is the same both times and
the ledger holds the winner's row alone. That is the claim the branch-and-bound refutation left
behind, *the effects a cancelling shape leaves depend on the choice and its answer does not*, and
a schedule cannot reach it because a schedule cannot decide which branch wins.
`tests/test_race_shapes.py` carries both directions; a loser that is not stopped writes both rows
and reddens all four cells.

## What a held clock forces, which is the cancellation a schedule could not reach

The reason a race has no `Shape` row is that nothing can decide its winner. A **deadline** is a
different trigger for the same cancellation, and it is decidable: a race reads one clock, so a row
that holds that clock past the deadline decides the race before a branch is launched, since the
deadline is read before the first one runs. `TimedOut` names no winner, so every branch is a loser and the ordinary two
admission checks stop them. Nothing about the schedule enters.

| the row wants | it holds the clock |
|---|---|
| every branch stopped before it runs | at or past the deadline, which the race reads before it launches one |
| a branch to land on a named instant, and the tie decided at it | before the deadline, and the branch moves it as it ends |
| the deadline never to arrive | before it, an hour off, which bounds each wait and orders nothing |

Holding the clock decides WHETHER the deadline has arrived; it does not make time pass, so a wait
the clock never reaches is a wait in real seconds. One property needs that time to pass and cannot
be held still: that the deadline WAKES a race whose branches are all running. Its row puts the
deadline a breath ahead of a running clock and blocks both branches until the choice lands, so the
timeout wake is the only way the race can move and nothing is ordered by the interval.

`tests/test_race_deadline.py` is the worked set, and every mutant planned against the race's
deadline path dies against it.

## The row a held clock made possible

`tests/test_watched_descent.py` is the first `Shape` row that cancels: a linear drill whose every
level races its branches against one instant, spelled by `unfold` and by `fix`, on both engines,
crashed at every checkpoint. The clock is held for the whole row and each level stamps it at its
own depth, which is where the cut lands.

| the check | what it says |
|---|---|
| `agree` | the two spellings answer alike, place the same names and write the same rows |
| `sweep` | a crash at every checkpoint converges, the race's choice and its endings included |
| `sweep_pairs` | two crashes converge: each checkpoint, then each later one on the resumed attempt |
| `interleave` | the descent is schedule-independent, predicted before it ran, and every run reaches its answer |
| `stopped` | the endings name the cut level's branches and no other |

**A level's stamp is absolute rather than an increment**, and that is what makes the answer
converge: a stamp says where the clock stands at a depth, so re-running one is idempotent, and
SKIPPING one, which a retry does for every level whose step is already recorded, leaves the clock
where the crashed attempt put it, which is where the level being resumed needs it. An increment
says how far the clock has moved, so it carries a discarded attempt's count into the retry, and it
reddens the four sweep cells and nothing else. The convergence holds under this clock's evolution
and no other: a crash before a choice commits leaves that race to be decided again, and a clock
moved to the bound in between would legitimately cut it there instead.

**Where the row's edge is.** Its path is `deadline_of`, `Racing.expired`, `before_any_branch`,
`decide` with `expired`, the stored choice and the loser's stop. Everything downstream of when a
branch ENDED is outside it, and four mutants mark the line: `in_time` answering True,
`concurrently`'s wake bound dropped, `in_order`'s end stamp lost, and `read` dropping the
late-success evidence each leave this file green. `tests/test_race_deadline.py` pins the tie and
`tests/test_hedge.py` turns on `in_time` directly, since its timer's own step leaves the clock past
the bound and only `in_time` ranks that success late.

**A schedule cell holds each run to the row's answer.** `interleave` applies `agree`'s checks to
every run of a row that carries an `answer`, since independence alone passes over a degenerate
run: a descent that runs to its budget is as schedule-independent as one that is cut. With the
race's deadline removed from the substrate, the descent's four schedule cells go red through the
harness, where independence alone left two green.

**Two crashes reach what one cannot.** `sweep` crashes a run once, so an attempt past the second
is never reached. `sweep_pairs` crashes a pair of checkpoints in two ATTEMPTS: the second aim arms
only when the next attempt starts, since a gather sibling still running in the crashed attempt
would otherwise take it, and every run is held to one attempt more than its crashes. A pair is
tried in both orders, because a store's listing is not commit order under a clock that steps back,
and only two branches of one gather or race may go uncrashed, since one attempt commits both past
a crash in either. An engine refusing checkpoint writes on a third attempt leaves `sweep` green
and `sweep_pairs` red. The pairs grow as the square of a run's checkpoints, so a row opts in: the
descent, the tree search and the pruned search do. `tests/test_sweep_pairs.py` pins the helper.

## The allowance, and why it is not a workaround

A cancelling shape cannot use its order up, so `Turnstile.check` takes `stopped`: the ops it may
legitimately never see. That set is **read off the run's own endings record**
(`_shapes.cut_short_by_a_race`), never counted by the test author, so the instrument's expectation
is held against the substrate. A branch that `won` or ended `unchosen` reached its last op and
every one of its ops must have arrived; `refusal`, `stopped` and `raised` each end a branch where
it stands.

The allowance names the BRANCH and not the op, because how far a stopped branch got before the
choice landed is exactly what an interleaving varies. A branch carries every frame enclosing its
race, a scope's or an outer race branch's alike, since an `endings` record always sits under its
own race's frame: the hedge's inner race is `race:0,1;race:0;endings`.

**How a row fires it.** A branch has to be cut with an op the schedule NAMES still ahead of it, so
the choice must land before that op's admission, which a loser meets before any layer does. A
turnstile alone cannot arrange that, since it releases the next turn from its layer's `finally`,
before the winner's body returns; the choice hold can, and the two coexist when the call that
waits for the choice is OUTSIDE the order. An
ordered call that waited would keep its turn and deadlock the schedule; an unordered one holds no
turn at all. `tests/test_race_shapes.py` carries the row under both orders of the branches' first
steps, and checks the allowance both ways: taken it passes, withheld it names the op nothing
allows. A turnstile that forgave nothing reddens those four cells and no other test in the tree.

The watched descent declares the same reader and forgives nothing, which is correct rather than
vacuous: it cuts its level before a branch yields any op at all.

## A release from inside the store write

Some windows sit between two writes one thread makes, where neither a schedule nor a clock reaches:
`RaceState.publish` stores the choice, then puts it in each cursor's bounds. A test reaches that
window by making the store write itself the release. The `settle` it hands `publish` sets an event
a loser thread waits on, then waits until that thread has either ticked or started waiting on the
seal, which a `Condition` whose `wait` announces itself makes observable. No sleep orders anything,
and without the seal the loser runs past the saved cut every time
(`tests/test_race_publish_seal.py`).

## The corollary

[[concepts/enforcer-domain]] asks what a gate SCANS. This is its twin one level over: **ask what an
instrument can FORCE.** A schedule that runs green over a race is not evidence the race followed
it, and the failure is silent in the useful direction, since the order the index already agrees
with passes every time. The move that finds it is the same one: reverse what the instrument
describes and see whether the verdict follows.

The general form is a forced input, and its observation-side twin is two hypotheses with one
reading: there two hypotheses look alike THROUGH the instrument, here the instrument sets something
the outcome does not depend on ([[concepts/evidence]]).

Some race and quorum shapes are still unwritten; [[concepts/recursion-shapes]]
classifies them, and its "How a shape is tested" holds for every shape but a cancelling one.
