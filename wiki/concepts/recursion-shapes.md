# Recursion shapes, classified by what they make the runtime answer

`effective/combinators.py`.

**A classic shape is a semantic probe.** Run through the substrate, it makes identity, replay,
governance, spend and joins answer the same question at every node, so a boundary that does not
compose shows up as a disagreement. Recursion is the forcing function; the defects it finds are
rarely recursion bugs. Repeated brackets are where spawn names, ledger rows and approvals alias,
and tree search reaches that shape by design.

**What the nesting reaches, and what it misses**, on the coder:

| a nested run under a scope with a shared run id reaches | it misses                                       |
|---------------------------------------------------------|-------------------------------------------------|
| names that alias across levels                          | state held handler-side across attempts         |
| values that must climb, such as a child's conclusion    | anything at a task boundary                     |
| coordinates that must compose, depth by depth           | anything at a park                              |
| a refusal that must carry the run it stopped            | crash resume, until the probe sweeps crashes    |

**A probe reports only what its inputs leave honest.** The coder's first Y test passed its depth
counter in the visit slot of `Under`, and recursed cleanly; read off the `Ctx` instead, the same
coordinates collided at depth 2, which is what made the placement a sequence. `fix` is the showcase:
the collision it is credited with was found first by two plain nested calls.

Shapes are catalogued on two axes: the **topology** a shape adds to the induced graph, and the
**error kinds** it needs the substrate to distinguish. A shape earns a combinator when a caller
needs it; the rest stay rows here.

## Topology

| topology it adds | shapes | status |
|---|---|---|
| sequence | linear descent, pipeline / transducer, the visits of a state machine, the turns of a ReAct loop, mutual recursion in one task | `descend` exists; `unfold` spells it in-task; `run_machine` and `run_agent` are `descend` judges, a visit or a turn per level under `d:{n}`; `mutual` runs roles that hand off to each other in tail position, a hop per level under `d:{n};state:{role}` |
| branch and join | divide and conquer, fork/join, map/reduce, AND/OR search | `recurse` exists (map/reduce); divide and conquer through `unfold`, in-task or across tasks |
| repeated bracket | tree search (MCTS), minimax, beam search | `tree_search`, rounds of `unfold`; `search.mcts` over it, and `search.beam` as a `search.frontier` search, itself a `descend` judge. Rounds refill through a `round-grant` park, and a budget refusal ends a search with what it has whole: `tree_search` its earlier rounds, `beam` its last scored frontier. A refusal while the roots are scored has no frontier to answer with, so `beam` raises it, where `tree_search` in the same position returns its initial state |
| cycle to convergence | fixpoint / iterative refinement, worklist / chaotic iteration | `fixpoint`, a `descend` judge answering `Converged` or `Unconverged` with the budget that stopped it; a worklist is `fixpoint` over what is pending and what is known, whose pass must make equality mean nothing is pending |
| sharing | memoized recursion / dynamic programming | classified |
| cancellation | race / hedge, quorum, branch and bound, best-first | `race` and `quorum` are built, with cooperative cancellation inside them and no cancel op. Branch and bound is built with no cancellation, its bound threaded through the recursion's own state ([`tests/test_pruned_search.py`](../../tests/test_pruned_search.py)); as a race it is refuted. Best-first is built as a `search.frontier` policy that picks the top k, and as a race it is deferred. Meets reversal in one cell, below |
| reversal | saga / compensation | classified, and **forced rather than chosen**: the ledger trigger forbids UPDATE and DELETE, so a committed row is undone only by a later row |
| cross-task | mutual recursion across tasks, supervisor tree, blackboard / agenda | mutual recursion across tasks deferred |

**Cancellation and reversal are orthogonal, and they intersect in one cell.** Cancellation asks how
a loser learns to stop and when it is quiescent; reversal asks what forward event restores the
invariant. A race whose losers appended nothing needs the first and not the second, and a saga
failing at step 5 needs the second with nothing concurrent to stop. The cell where both apply is a
**cancelled branch that already committed**, and it is the only place cancellation raises a question
about interleaving, because whether the loser committed is a fact about the schedule.

Two things follow, and they point opposite ways.

| against the crash case | for it |
|---|---|
| the cut is **voluntary**, so after quiescence a canceller READS what committed instead of reasoning over every cut the way [`docs/effective-design.md`](../../docs/effective-design.md) §10's crash-preservation row must | the signal **races the loser's progress**, which a crash does not: a crash stops everything, a cancel asks, and the loser may commit while the request is in flight |

So the footprint to compensate is not known when the cancel is issued, only once the loser has
stopped. That is the discipline the meter already uses one level down: a branch's spend reaches the
root when the barrier folds it, never before ([`docs/effective-design.md`](../../docs/effective-design.md) §9.4). **Compute a
compensation at the barrier, not at the signal.**

Nothing here is a design choice. Rollback is unrepresentable, so a compensating append is the only
recovery available; and a cancel that unwound a captured stack is exactly what the no-`call/cc`
invariant rules out, so compensation over unwind is forced from both ends. The substrate ships the
control half already, as `Refused` routed around as a value; the compensating event is the domain's.

Sharing is the inverse of the aliasing defect: two paths *should* reach one durable result. A
naming rule where placement always separates footprints cannot express it, so a sharing combinator
names its subproblem by a semantic key the author supplies. Settling many grants or approvals with
one answer, an inbox over parks and agents on a message board are the same case; they wait on one
design, not yet taken.

## Error kinds

| kind | substrate form | behaviour | shapes that need it |
|---|---|---|---|
| runtime refusal, an answer about the world | `Refused`, `BudgetRefused`, `ToolRefused` and a refused spawn, recorded as values; `ChildRefused` from a joined child; `RunRefused` from a nested machine, carrying the uncommitted run for `committing` | routed around in the workflow, including out of a `scoped` body or a `gather` round once every leaf is a refusal; a loop re-raises a `BudgetRefused`; replay serves the recorded answer; a child completes and answers `Refusal` | backtracking, quorum, saga, governed search |
| crash, transient | an uncaught exception | retried by the engine to `max_attempts`; on the last attempt the task fails and its worker answers the parent `Failed` | supervisor tree |
| programming error, deterministic | `CompositionRefused`, an `ops.Unretryable` | its task fails on one attempt, reported unwrapped, and its worker answers the parent `Failed` | any shape whose author reuses a name |
| child failure | `ChildFailed` from a joined child, also `Unretryable` | caught, the parent supervises; uncaught, the parent fails once and the failure climbs, each task naming its child | supervisor tree |

A retry of a programming error reaches the same line and fails the same way, because replay makes
it deterministic; one among a gather's errors is enough, and every branch's error is reported. A
worker that dies on a child's last attempt is the one ending no parent hears. Cancellation, which
race and quorum need for their losers, is a fifth behaviour with no substrate form.

## Tail transitions

A composed state nests only while a component has work left after its child returns: `yield from`
composes machines hierarchically, and each delegation that has more to do holds a frame. That
continuation is the part of the product state a run actually builds, so what bounds depth is
whether a transition is a tail call.

| transition | combinators | live state as depth grows |
|---|---|---|
| tail, in-task | `Deeper` in `descend` and `unfold`; the rounds of `tree_search`; `fixpoint`'s steps; `Hand` in `mutual` | constant |
| tail, across tasks | `respawn`, whose generation boundary ends the task | constant per task |
| continuation kept | `Branch` and its join; `recurse`; recursion through `yield from`; nested `scoped` | a frame per level, and a thread per `Branch` level |

Rewriting a recursion buys no depth; cutting its continuation does. A tail transition is one whose
outer level has nothing left to do, so the driver drops that level and loops, and the run-time state
stays the size of one level's state. `Deeper(narrowed)` is that tail call returned as data.
[`tests/test_descend_depth.py`](../../tests/test_descend_depth.py) runs a descent past the recursion limit on the recorder, replay and
both engines, beside a `yield from` recursion that raises at the same depth. The composition claim
this bounds is that the product of composed state machines exists in the semantics and not in the
source.

A shape belongs in the first two rows when its step hands the next step everything it needs. A
ReAct turn, a coding-machine visit, a fixpoint iteration and a worklist pass do. A step that uses a
child's result afterwards, as a join does or a turn that runs a subagent in-task, is in the third.

## How a shape is tested

A shape is a row of [`tests/_shapes.py`](../../tests/_shapes.py)'s `Shape`: its spellings, one of them written with `fix`,
the domain they run against, and the oracle every run must meet. `agree` runs each spelling and
compares their answers, ledger rows and checkpoint names; `sweep` crashes one spelling at every
checkpoint and holds each crashed run to the same oracle. The `backend` fixture runs both on each
engine.

**A cancelling shape is the exception, and it is not a gap.** `race` and `quorum` have one
spelling, a crash reschedules them so `sweep` has nothing to hold constant, and a forced schedule
cannot decide a race's winner at all: [concepts/forced-schedules](forced-schedules.md) has the measurements. They are
pinned by a two-event hold instead, in [`tests/test_race_shapes.py`](../../tests/test_race_shapes.py).

A reference spelling reads for itself every decision it checks: how many levels a grant gives, and
where a refusal is caught. A reference that copies the combinator's structure agrees with it on a
defect, as a shared `refill_levels` and a grant ask placed outside the catch each showed, so each
shape also keeps rows whose answer is written down.

## How the combinators run

`unfold` over a node decision `Answered | Deeper | Branch`, and `descend` is `unfold` with a judge,
a node that never branches: `Deeper` trampolines, so a linear chain does not grow the stack; `Branch` runs its
children as a `gather`, which keeps per-branch child
handlers, concurrency, the per-branch meter fold and per-branch crash-resume. A driver-owned join
stack was measured and not adopted: it buys depth only when a node's name stops being a nested
scope, and for a shape that fans out the bound is node count. A chain of single children is
`Deeper`: each `Branch` level holds a thread and a nested gather, so a one-child chain meets the
process's recursion limit: at the default limit of 1000 it fails with `RecursionError` on SQLite
within a few hundred levels, and raising the limit raises the depth. `fix`, the typed closure
fixpoint, is the showcase and a test; the literal Z form has no type. A node that does not answer at
its final level raises `DescendedPastBudget`, a composition refusal.

The budget counts levels along a path. When a path exhausts it, `unfold` parks for a grant (a
`grantor`, or `run_id`'s built-in park) outside the level's scope, so each child of a `Branch` that
exhausts parks on its own. Collapsing those decisions is the caller's policy: a rules or agent tier in
`grant_cascade`, or an allowance granted once above the fan-out. Branches that park together are
answered one wake at a time: an answer to a branch other than the shown park waits until that park is
answered, and a batch of answers finishes the run in one drive.

A `Branch` across tasks (`crossing=AcrossTasks`) cannot be a `gather`: a done-event await inside a
gather branch is refused, so its children are spawned in a loop under `rec:{i}` and joined in a
loop. A child's levels are its spawn depth, so a node's final level is where the handler would refuse
its spawn, and a refill does not cross tasks: it grants levels and no spawn depth. Each child runs
`unfold_task` under `spawning.run_child`, which answers the parent with the value or the refusals
the child stopped at; the parent re-raises them as `ChildRefused`, so a refusal climbs to the root.
A child that fails for good is answered `Failed` by its worker, and `ChildFailed` climbs the same
way. A task body reads its attempt as `ctx.attempt` on either engine.
