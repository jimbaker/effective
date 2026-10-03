# The machine is an extended finite state machine

`src/effective/machine/`, `src/effective/coding/transition.py`.

A state has **one slot** (`run: Run[S, V, R]` returning a `Report`), and the worker/judge pair is
`specs.fuse`, a construction an embodiment may decline. A judgment that needs an address of its own
(a budget line, an edge in the table, a name a pack can bind to) is written as a state; everything
else fuses, which is the right default. Every judgment gets a `Ctx`, so needing coordinates is no
longer a reason to split.

## Two invariants, and they are refusals rather than preferences

**The transition never reads the register.** `transition(state, verdict)` has a finite domain
precisely because guards are precomputed into the verdict, and that is what lets `ty` narrow an
enum match to `Never` and makes a dropped arm a type error. A textbook EFSM with guards over
registers has an unbounded domain and loses totality checking outright. A request for
`transition(state, verdict, acc)` is refused with that argument, not weighed against it.

**The core never learns about skills.** A binding is not a substrate concept and `StateSpec` gets
no field for one. Skills compose at the *embodiment* layer, as a combinator over a `Run`, so a pack
names a state: one slot, not two.

## The one register

`carried` belongs to the commit protocol: content-addressed, handed to the predicate tool, its
**keys** becoming the committed file list. So there is no *uncommitted* register, which is why an
optimizer's fold is not expressible yet.

## Durable suspend is replay, never a captured continuation

No `call/cc`. A durable suspend/resume re-runs the deterministic computation and replays recorded
op results; a layer-injected suspend resolves on replay rather than by resuming a captured layer
stack. The `yield from` effect surface is the whole point: a generator is a **delimited, one-shot,
explicitly marked** suspension, the tame and useful sliver of continuations with none of `call/cc`'s
arbitrary-capture or multi-shot hazards. Thread state in from *recorded* state.

`call/cc`, like software transactional memory, is a beautiful abstraction that makes real systems
harder. Reach for explicit, restartable mechanisms instead. This is a rejection, not a pending option.
