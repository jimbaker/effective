# ADR-0024: The walk: one mint, many interpreters

- **Date:** 2026-08-13
- **Status:** Proposed for the unified walk (§5), which is designed and not built. The preparatory
  steps are built: `handlers.base.walk_run`, `layers.layer_routing`, total dispatch in the replay
  and recording handlers, `ReplayHandler`'s `Respawn` arm, and `keys.gather_prefix` (§4). A
  principal's steer (§8) is built as `effective.steering.SteeringCtx`, below the walk.
- **Relates to:** [ADR-0020](0020-key-composition-one-grammar.md) (key composition; its gather-prefix trigger is what §4 discharged, and
  it owns `FramePosition`), [ADR-0008](0008-dynamic-workflows-as-ops-applicative-parallelism.md) (gather as structure, whose branch coordinate is the frame
  this ADR unifies), [ADR-0009](0009-durable-backend-two-regimes-taskcontext.md) (the two engines whose conformance suite is one axis of the safety
  net; [`tests/_walks.py`](../../tests/_walks.py) is the other), [ADR-0021](0021-dynamic-graphs-as-projections.md) §6 (promote-a-fork-to-canonical, which will not
  be built; §7 shows adding a `gather` branch is the same back edge), [ADR-0022](0022-dashboard-projections-the-read-side.md) §4a (the span join),
  [ADR-0025](0025-race-and-quorum.md) (race, the second structural op with branches).

## 1. The decision

**Every interpreter of the op language is a legitimate handler; every re-implementation of the
WALK is not.** The walk is op dispatch, frame descent and ordinal placement. It becomes one shared
mechanism that interpreters drive, reached in three sequenced steps: the first two are
independently valuable, and the third is what they add up to.

Many interpreters is the thesis and is not in question. One op language with many handlers is what
algebraic effects buy: the loops differ in where a leaf's result comes from, how they park, how
they run branches, and whether layers see the op, and each of those is a handler's proper business.

## 2. The count

Six loop bodies drive a workflow and assign identities. Five are enrolled in
`tests/_walks.py:ALL_WALKS` as six walks (the durable loop runs once per engine); one is enrolled
nowhere.

| walk | loop | enrolled |
|---|---|---|
| `RecordingWalk` | `RecordingHandler._drive` | yes |
| `ReplayWalk` | `ReplayHandler._drive` | yes |
| `SqliteWalk`, `AbsurdWalk` | `DurableHandler._drive` | yes |
| `LiveForkWalk` | `fork.live_drive` → `_live_loop` | yes |
| `MeasuredForkWalk` | `fork.measured_drive` → `_MeasuredRun.drive` | yes |
| none | `fork.replay_prefix` and `_replay_scoped`, reached from `fork_at` | no |

The census is the seven `with placing(` sites in `src/effective`: one each in the recording, replay
and durable handlers (`src/effective/handlers/`), four in [`src/effective/fork.py`](../../src/effective/fork.py). [`tests/test_walks.py`](../../tests/test_walks.py) buckets coverage by module,
so it cannot see that `fork.py` holds an unenrolled walk beside two enrolled ones: a gate is
bounded by what it scans, on the census itself.

## 3. Why: the extraction has been happening one defect at a time

`walk_run`, `layer_routing`, `gather_prefix`, `refuse_respawn_in_branch`,
`refuse_a_park_in_a_race_branch` and the shared `placing` mint each exist because a walk diverged
or was about to. The walk is being unified incrementally, and each increment was paid for after
the fact.

The counter-argument is the reason this is sequenced rather than immediate: [`tests/_walks.py`](../../tests/_walks.py)
already asserts the enrolled walks agree. Its own docstring states the limit: it catches
divergence and is silent on a defect that is uniform across walks. The hand-spelled gather frames
that §4 replaced were exactly that blind spot: they agreed with each other and would have been
wrong together.

## 4. Step 1: `gather_prefix` (built)

[ADR-0020](0020-key-composition-one-grammar.md) set the trigger for a `gather_prefix(branch)` sibling to `scope_prefix`: a sixth mint, or
any reader that has to learn the shape. Both fired: there were six hand-spelled mints
(`absurd._branch_handler`, three in `absurd._join`, `recording.run_branch`, `replay._drive`), and
`graphview.strip_branches` is the reader.

`keys.gather_prefix(g, i)` is the fragment form, composed by `compose_key` and ending in the term
separator, and every handler site calls it. `race_prefix(r, i)` is its sibling for a race branch
([ADR-0025](0025-race-and-quorum.md)). `api.gather_frame` stays separate because it wraps a finished key; the handlers need
the fragment. `frame_path` cannot take a gather, since `op_key(Gather)` raises and there is no atom
to pass.

## 5. Step 3: the unified walk (designed, not built)

The substrate's own thesis applied to itself: reify the walk as a generator yielding typed
descriptions, and let a swappable interpreter drive it. The proposed shape, in which `walk`,
`Placed` and `EnterBranches` are names this ADR proposes and do not exist:

```python
def walk(
    gen, frame: str = "", *, in_gather: bool = False
) -> Iterator[Placed | EnterBranches]:
    position = FramePosition()      # a stack local: nesting restarts at 0 by construction
    ...
    match op:
        case Gather(branches=branches):                      # structure: no entry, recurse
            g = position.next_gather()
            frames = tuple(gather_prefix(g, i) for i in range(len(branches)))
            send_value, throw = _unwrap((yield EnterBranches(g, branches, frames)))
        case Race():                                         # structure, with its own ordinal
            ...                                              # position.next_race(), race_prefix
        case Scoped(scope=scope, body=body):                 # structure: the walk composes
            send_value = yield from walk(
                body(), frame_path(frame, scope), in_gather=in_gather
            )
        case Respawn():                                      # terminal: no entry, ends here
            return _boundary(op, in_gather=in_gather)
        case Step() | AwaitEvent() | AppendLedgerRow() | StoreArtifact() | SleepUntil():
            with placing(op, position):                      # leaf: one entry, one placed key
                key = placed_key(op).prefixed(frame)
            send_value, throw = _unwrap((yield Placed(op, key, frame)))
        case unreachable:
            assert_never(unreachable)
```

The dispatch is a `match` closed by `assert_never`, the form the replay and recording handlers
already use, so a new arm of `WorkflowOp` is a type error rather than a fall-through. `in_gather`
is a real parameter rather than inferred from a non-empty `frame`: a `Scoped` body carries a frame
and is no kind of branch.

**Measured:** a prototype walk minted keys byte-identical to the recorder and replayer across a
leaf, a `scoped`, a two-branch `gather`, gather-in-scope (ordinal restart and continuation),
scope-in-branch, repeated step names, two sleeps, and repeated awaits with occurrence-suffixed
settlement keys.

**Not measured:** park, refusal delivery, concurrency, layers, the durable ctx swap, `Respawn`,
race, and recursive driving of nested branches. The prototype's interpreters were toy
leaf-functions.

What the protocol carries:

| the walk owns | stays interpreter policy |
|---|---|
| key and frame minting, once | layer routing (consulted per walk) |
| the structural descent (`Gather`, `Race`, `Scoped`) | `placement_scope` (durable only) |
| the ordinal, on the Python stack | park, in three mechanisms |
| the answer protocol (§6) | absolute-await addressing |
| | what the gather barrier folds |
| | refusal delivery |

## 6. The answer type: provenance and authority are one field

An interpreter answers the walk. The answer says how it was obtained, so the vacuity taxonomy that
[`scripts/vacuity_probe.py`](../../scripts/vacuity_probe.py) reverse-engineers at the store seam becomes a field:

| answer | the key's role | who decided |
|---|---|---|
| `Computed(v)` | inert: written, never consulted | the interpreter |
| `Resolved(v)` | selective | the store, by key |
| `Matched(v, at)` | witness: position chose, the key checked | the trace |
| `Steered(v, by)` | selective | a principal, by key (§8) |

The mechanism forces the provenance: a replay interpreter cannot return `Resolved`, because it has
no store to look in, and a durable one cannot return `Matched`, because it has no position. The
probe attaches only to `SqliteTaskContext`; an answer that carries its provenance makes every walk
one observation point. None of the four constructors exists yet; `Steered` is built with the other
three, since adding it later means revisiting every interpreter's answer site.

## 7. What is refused, and why each looks reasonable

A branch set is steerable in proportion to how much of it is data. Where branches are code you can
only answer them; where the set is data you can change it.

| refused | because |
|---|---|
| drop a `gather` branch | the result list is positional; it shrinks and every downstream index shifts. A type change, not an override |
| renumber survivors after a drop | an index is a coordinate: unmodified `gather:0,2;step:b2` would become `gather:0,1;step:b2`, an identity nobody steered |
| add a `gather` branch | a closure is not in the tape: the program yields 3 branches, the tape holds 4, and the fourth has no body to re-execute. [ADR-0021](0021-dynamic-graphs-as-projections.md) §6's back edge in another form |

Answering a branch (supplying its result without running it) is type-safe and key-stable, and is
replay-safe by design once the answer is recorded at the branch coordinate, which is unbuilt.
Adding a hypothesis to `fork.marginal_sweep` is already expressible, because its
`deltas: Mapping[str, Substitution]` is data.

## 8. Steering

A steer is a value a principal supplies for an op the run has not reached: at this key, answer V
rather than calling the domain. A steering point must be quiescent, addressable and resumable,
which is the delimited, one-shot, explicitly marked suspension the no-`call/cc` invariant already
commits to.

- **A steer preserves `replay(tape) = tape` only if it is in the tape.** Out-of-band mutation
  breaks the fixpoint as `call/cc` would, by another door.
- **Steerability follows the bookkeeper.** A checkpoint is disposable, so substituting its value is
  a re-execution the substrate already supports. The ledger is canonical, so steering it would be
  rewriting history, and the append-only trigger refuses it.
- **A steer must satisfy the declared result type of the op it answers** (`CallTool.result_schema`,
  `AskLLM.response_schema`), which the walk can check because it holds both the key and the op.

What is built is `SteeringCtx`, a `TaskContext` stacked around the ctx a `DurableHandler` drives,
the way `run_fork` stacks `SeedingCtx`. It answers a steered `step` by handing the inner ctx a thunk
that returns the steer's value, so the answer commits through the ordinary checkpoint path,
atomically with the step, on either engine. Steers are keyed by the occurrence-resolved key
(`name#k`); `applied` reports which fired and who authored each, and the caller records who
decided as a ledger row, so no checkpoint encoding changes. `unapplied` is the completeness check.

The ctx sees `(name, thunk)` and never the op, so two checks stay with whoever authors the steer:
the declared result type, and the raw checkpoint encoding of `Steer.value`. Only `step` is steered; an
await's answer arrives by `emit_event` and is a different mechanism. The walk-level `Steered`
answer (§6) is what would let the walk check the schema itself.

## 9. Sequence

**Built:**

| piece | what it closed |
|---|---|
| `handlers.base.walk_run` | the per-run ambient (`enter_task_run`, `run_scope`, no meter) in one place for every walk |
| `layers.layer_routing` | the three layer-visibility tuples as one total decision; [`tests/test_layers.py`](../../tests/test_layers.py) pins that they are disjoint and cover every arm |
| total dispatch | the replay and recording tables end in `case unreachable: assert_never(unreachable)` |
| `ReplayHandler`'s `Respawn` arm | a `Respawn` has its own arm: it ends the run at the generation boundary, and is refused inside a gather branch |
| `keys.gather_prefix` | step 1 (§4) |

**Proposed, in order:**

1. **Frame state on the stack.** Replay already passes `prefix` and `position` as parameters; the
   recorder and the durable handler keep them on the handler (`self._prefix`, `self._position`).
   Two sites cannot simply drop that state: `recording.ScopedSuspended`, which re-installs its
   `prefix` around each resume, and the durable handler's ctx swap. A stack local cannot survive a
   park, so this is stack state while live plus an explicit resume record while parked.
2. **The unified walk.** Build `walk()` and its protocol, with [`tests/_walks.py`](../../tests/_walks.py) extended to assert
   it agrees with every enrolled walk before any handler adopts it, and `replay_prefix` enrolled.
   Then migrate `ReplayWalk` (no layers, no park, smallest), `RecordingWalk` and the durable walk,
   then the fork drivers. Each old loop is deleted only when its walk passes conformance.

**Skipped: extracting the generator pump.** It is identical in three loops and has never been a
defect source; it disappears inside the unified walk anyway.

## 10. Consequences

**Step 3 touches the operational semantics**, where a defect is a durability defect. The loops are
small because the extractions in §3 happened, so unification removes little code and adds a
protocol whose control flow is harder to read than `while True`. The per-loop comments are an
asset and must survive the move: `RecordingHandler._run_scoped` drives a scoped body with `_drive`
and not `_run` for a recorded reason. Unification also prices in a parameter for every future
divergence, and the engines differ where it counts: an Absurd park writes a null-payload marker row
into its event table where SQLite writes none.

**A named divergence must stay expressible.** `LAYERED_OPS_DURABLE_ONLY` is deliberate: the
recording core decides an `AwaitEvent`'s park or resume before the layer stack, because the
no-`call/cc` invariant forbids parking mid-layer. A unified walk is where someone would be tempted
to remove it.

**Totality is an obligation of the walk's dispatch.** One walk makes the arms total only if its
`match` has no leaf catch-all.

**A handler nothing drives rots.** `ReplayHandler` has no caller in `src/`; every construction is
in `tests/`. A handler the substrate never drives is one whose arms nobody was forced to complete,
which is how it lacked a `Respawn` arm and how `replay_prefix` stayed unenrolled. That argues for
unification, and warns that whatever the walk leaves optional will rot the same way.

**Two design questions come first.** What `Placed` carries across the yield (a held `placing`
context and coordinates on `Placed` differ observably, and the protocol must pick one), and how
`Respawn` appears in the protocol.
