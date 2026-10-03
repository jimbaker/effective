# What each watcher watches

Four things in this substrate watch a running computation, and the common names for them come from
different fields, so they were converging on one word. They are split by **what each one watches**.

| the name | watches | acts by | where it lives | the field it is named from |
|---|---|---|---|---|
| `supervisor` | a child's ENDING | catching `Returned \| Refusal \| Failed` and recovering, or letting the failure climb | a spawn tree, task to task | Erlang/OTP, where a supervisor's job is a child's lifecycle |
| `watchdog` | a running body's PROGRESS | cutting the body before it ends on its own | in one task, around a recursion | distributed systems, where a watchdog fires on lack of progress |
| `monitor` | a TRACE of arrivals from outside | deciding a property over it and answering | the arrivals bridge, one subscription | runtime verification, where a monitor decides a property over a trace |
| `deadline` | nothing: it is the BOUND the other three may carry | expiring | an absolute instant on an op | gRPC's distinction, a point in time against a duration |

**A supervisor reacts to a termination; a watchdog reacts to the absence of one.** That is the
whole difference, and it is why they cannot share a word: the spawn tree already hears endings
([[concepts/recursion-shapes]]'s error kinds), and nothing yet cuts a body that is still going.

**A monitor decides, and does not act.** LTL₃'s three-valued verdict is its shape, and the arrivals
protocol reached the same union from the other side: `Satisfied | TimedOut | Incomplete`, each
carrying the decision it was recorded under. What a caller does with the verdict is the caller's.

## Two words this split gives back

| the word | means | and does not mean |
|---|---|---|
| `observe` | ReAct's third step: what the TRANSCRIPT sees of a tool's result, against `result`, which is what the op carries and the workflow folds | a name for an op that watches a run, and no op does |
| `telemetry` | spans, the disposable third bookkeeper, in the OpenTelemetry sense the rest of the industry uses | something a substrate primitive competes with |

No op watches a run. Reading disposable state and emitting nothing durable is what telemetry
already is, and a second mechanism for one job is the redundancy this vocabulary exists to stop.

## What the deadline reaches, and where the two verdicts stop agreeing

A race answers `Chosen | TimedOut | Impossible` and a monitor `Satisfied | TimedOut | Incomplete`.
Two of the three line up, and the third pair does not: `Impossible` is a fact about the BRANCHES,
the quorum can no longer be met, while `Incomplete` is a fact about the INSTRUMENT, a source or
the protocol failed to cover the trace. Only a trace arriving from outside can have one, which is
why the two unions stay apart rather than converging.

That asymmetry says which watchers the deadline alone unlocks.

| a monitor over | can reach | needs |
|---|---|---|
| an INTERNAL trace, the run's own durable progress | `Satisfied` or `TimedOut` | the deadline, and no bridge |
| an EXTERNAL trace, arrivals from the world | all three | the arrivals bridge, for the third arm |

## How a watchdog is spelled

A recursion whose every level races its work against one instant, and whose judge reads the run's
own durable progress once a level is cut. `tests/test_watched_descent.py` is the worked row.

| the part | what carries it |
|---|---|
| the bound | one instant, read through a step before the descent starts and handed to every level |
| the cut | the race's timeout, which names no winner, so every branch of that level is a loser |
| the property | a step reading the LEDGER, so a crash leaves the reading intact |
| the verdict | the level's answer: what the trace holds, or the value the descent reached |

The judge reads the ledger rather than the domain's own call log, and the difference is a crash: a
log accumulates the discarded attempt's calls, where the append-only record is what both attempts
agree on. A reading taken from the log passes a clean run and reddens every sweep cell, which is
the two-bookkeepers rule arriving as a test result.

## How a hedge is spelled

A bounded race inside an UNBOUNDED race's branch. `tests/test_hedge.py` is the worked row.

| what a hedge needs | what carries it |
|---|---|
| a second attempt starting at an instant | the inner race's deadline, whose `TimedOut` is the start signal |
| a first attempt still running when it does | the outer race, which names no deadline and so cannot make it a loser |
| the first attempt cancelled when the second answers | the outer choice, which stops it at its next admission |

One race with a deadline cannot be a hedge, and that is the shape worth stating: a race at its
deadline names no winner, so its first attempt is a loser and stops, which makes racing a second
one after it a retry. Putting the deadline on an INNER race moves it off the first attempt, and
the timer that expires is a step rather than a park, so the refusal of a parked race branch never
comes into it.

**The limit is the interpreter, not the primitive.** On a ctx whose branches cannot overlap the
first attempt runs to its end before the hedge arm is reached, so the second attempt never starts
and the first always wins. That is `Racing.in_order`, the path a deployed Absurd worker takes, and
it is the same weaker promise a race's deadline makes on a sequential ctx.

## Why a deadline is a keyword and not a combinator

A race's timeout is an outcome of the same choice as its winners, decided by one authority. A wrapper spelled `within(...)` would say the opposite, that the bound composes
over any effect; `quorum(k, branches, deadline=at)` says it belongs to the race. The substrate
takes an instant wherever it takes a bound, which is what [[concepts/bounded-wait]] settles for a
wait that names one and the race's deadline keyword for a race. Both are built.
