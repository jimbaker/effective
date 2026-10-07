# The repertoire axis: declaration, resolution, record

**The name.** *Repertoire*, because "skill repertoire" is immediately legible. "Loadout" reads
instantly to an FPS player and sent a reader of this design to look up a definition, and a term that
costs a lookup is not carrying the axis. "Loadout" survives as an illustration word in its
mission-kit sense: a set chosen for a mission.

**Status:** design, with two pieces built. §6a's `Tool[A, R]` ships at [`src/effective/react.py`](../src/effective/react.py)
with a fourth field, `observe`. §3a's `op_key` refusal arm ships at `handlers/base.py`, so read §3a
and §9 where they list it as a build item as describing work that is done.

**A skill binding is not a substrate concept**, so it is neither a declaration field nor a
decorator: skills compose at the embodiment layer. The repertoire reaches the model through
`machine.agent_worker(needs=...)` and its `decide_for` factory.

**External material is comparison color, never a premise.** The Cordis paper (an unreviewed
preprint) and the DeepSeek Harness repository appear only as comparison; no ruling below rests on
them.

---

## 1. What the axis is, and what it is not

> **Declare the vocabulary a workflow may draw on, resolve it against what a deployment supplies,
> and record what was actually resolved.**

Two properties are claimed, and **neither is reachability**:

- **Prospective legibility**: an orchestrator, human or agent, can see what a workflow will draw
  on without reading its source.
- **Retrospective completeness**: the record says what was *available*, not only what was *used*.

**It adds no expressive power.** An Effective program can already evolve an arbitrary graph from
its tape: a dynamic graph's three growth axes, `route`'s uniform dispatch, `recurse`/`descend`,
`spawn_fork`/`join_fork`, `respawn`. Nothing here changes what a program *can do*.

**Naming.** This was drafted as "the capability axis" and as Effective's answer to a "third axis"
Cordis reifies. Both are retired. "Capability" collides with the object-capability tradition's
stronger meaning, unforgeable authority, which this is not, since a declaration is descriptive
and the permission cascade is what authorizes. And the ordinal framing does not survive contact
with the repo's own taxonomy, which already decomposes onto **four** targets (control / data /
derived / executable),
nor with the fact that the axis Cordis names is *revertible registration*, which §7 below declines.
**What remains is a set chosen for a mission, which is what a repertoire is.**

### 1a. Alphabet, not word, and the slack is two-sided

A declaration bounds the **alphabet** of ops a workflow may yield; the tape records the **word** it
produced. Order, count, branching, depth and which `route` branch won are all unconstrained.

This keeps intact the rule that a graph is projected rather than declared, since *"a declared graph is a second place the structure lives, and
a second place can disagree with the first"*, because an over-approximation of an alphabet cannot
*disagree* with a word, only fail to contain it, which is a checkable refusal.

**But the slack is silent in the other direction, and that cost must travel with the claim.** A
declaration listing tools that are never used lies to exactly the prospective-legibility reader the
axis exists for. Containment failure is loud; over-declaration is not. A declaration that drifts is
worse than none, because it is read on trust.

## 2. The debt this pays: tunable seams take data, arriving for the third time

The strongest reason to build any of this does not depend on Cordis at all.

**Tunable seams take data:** a tunable seam's configuration must be *"a value the optimizer can hold,
diff, and re-derive… A closure is none of this: it neither serializes, diffs, nor mutates."* Today
a repertoire is `run_agent(decide=…, act=…, interrupt=…, compact=…)` and `route(handlers={…})`, all
closures. So **the repertoire is not yet a config**, and the premise that *"a loadout **is** a config"* is
aspirational rather than true.

That rule's own stated limit is *"not type-enforceable… a `Callable` parameter cannot know its
caller wanted it optimizable"*, with the mitigation *"a naming / **decorator** rule the seam author
opts into."* The decorator was the anticipated shape; this axis supplies the seam it was
anticipated for.

Three independent routes reached the same debt during the design: the repertoire, the alphabet, and
provision (§5). That convergence, not the comparison, is the argument.

## 3. Letters, and the alphabet's domain

A letter is an op's **`op_key`**, its identity *in itself*, never `placed_key`. Frames (a
gather's `gather:{g},{i}`, a scope's prefix) are **path**, not vocabulary.

The domain is forced by the substrate, not chosen. `op_key` (`handlers/base.py:288`) succeeds for
four arms and refuses the rest:

| arm | `letter` | why |
|---|---|---|
| `Step`, `AwaitEvent`, `AppendLedgerRow`, `StoreArtifact` | `op_key(o)` | content-keyed |
| `SleepUntil`, `Gather`, `Scoped` | ⊥ | *"identified by **where the walk found them**, not by anything the op holds"*: positional, hence path |
| `Respawn` | ⊥, **by ruling, §3a** | a task boundary, not a call |

**You declare what a workflow may *call*.** You cannot declare that it may fan out, sleep, observe,
or scope: those are structure, they are the path, and the path is projected, never
declared.

### 3a. Ruling: `letter(Respawn) = ⊥`, and continuation is budget-governed

`Respawn` is the ninth arm of `WorkflowOp` (`ops.py:272`) and `op_key` has **no arm for it**: it
falls through to `raise TypeError("unknown op")` (`handlers/base.py:350`). That is consistent with
how the rest of the substrate treats it: `layers.py:83` puts it in `UNLAYERED_OPS` with the note
that it *"orchestrates a TASK BOUNDARY, has no `op_key`"*, but as a *fall-through* it was an
accident rather than a decision.

**Ruled:** `letter(Respawn) = ⊥` by design. **Whether a chain may continue itself is governed by
the budget axis**: `Grant.add_generations` / `stop` (`budget.py:214`), answered by a human at a
durable park, **not by the alphabet.**

**The consequence must be stated, because it bounds the privilege ratchet.** §7's ratchet says a
self-rewriting chain may only narrow its alphabet, so it cannot escalate what it *calls*. It does
not govern the *door*: the alphabet has no way to say "may not respawn", and $A' \le A$ at a
boundary is today **asserted, not enforced**. Enforcing it is a handler obligation at the boundary,
not something the alphabet can express.

**Build item (not done here).** Give `op_key` a designed refusal arm for `Respawn` so the
substrate's line matches this ruling. **Subtlety that must not be missed:** the arm cannot raise
`ValueError`, because `placed_key` catches that and falls back to the walk's minted positional name
(`handlers/base.py:447-452`), which would invent a coordinate for a task boundary. The arm must
raise something `placed_key` does not catch, i.e. a *named* `TypeError`, turning today's silent
fall-through into a loud, documented refusal without changing behaviour.

## 4. The operational semantics

An extension of the small-step semantics in [`docs/effective-design.md`](effective-design.md), which it does not revise.

### 4a. Configuration: CLEAR

$$\kappa \;=\; \langle\, W,\; C,\; L,\; E \,\rangle_{A,\ \mathcal{R}}$$

**C**heckpoints · **L**edger · **E**vents · **A**lphabet · **R**egistry, with $W$ the continuation.
The design note already publishes $\langle W, C, L, E\rangle$ in that order; $A$ and $\mathcal{R}$ join as
**indices on the relation**, not fields, so the ratified tuple is untouched. The relation is
$\rightarrow_{A,\mathcal{R}}$ and composes with the scope index the semantics already uses,
$\rightarrow_{p_{g,i}}$.

Indices rather than fields because **no rule mutates either**: written flat, the bracket rule looks
like a mutation-and-restore; written indexed, there is nothing to restore.

**The partition, and the test that actually decides it.** Two questions, not one: the single
"which components appear changed in a rule's conclusion" test is insufficient, because *no rule in
the system changes $E$*:

1. *Does a rule change it?* → **state** ($C$, $L$).
2. *Can the environment change it inside one derivation?* → **state** ($E$); only between
   derivations → **context** ($A$, $\mathcal{R}$).

$E$ is state by (2): `peek_event` is a **non-suspending** probe whose own contract names *"an
emission landing between a branch's peek and the barrier"* (`handlers/base.py:470`). $\mathcal{R}$
is context under one stated assumption, **no hot reload within an attempt**, which holds here.

**Owed if $E$-as-state stays load-bearing:** the semantics contains no `Peek` or `Emit` rule, so
the partition is currently certified by an appeal *outside* the formal system. Two rules close it
(`peek` reading $E$; an environment emission extending it), and until they exist this section
states a property of the runtime rather than a theorem of the semantics.

**Derived, not carried:** the measured meter. Within a generation it is a left fold over $C$ (the
v1 `AskLLM` checkpoint value *is* `{result, usage}`: `_encode_usage_envelope`,
`handlers/absurd.py:1005`); a grant arrives through $E$; the cross-generation residue rides
`ACCRUAL_PARAM` on spawn params, absorbed into $W$ at `Task-Start`; canonical spend is a ledger
question. A separate component is a design the measured-spend accrual **rejected**, for crash-atomicity: a second
row is *"broken by a crash between the two commits."*

### 4b. The guard

$$
\frac{\;\mathsf{letter}(o) \neq \bot \quad \mathsf{letter}(o) \notin A \quad n \notin \operatorname{dom}(C)\;}
     {\langle\, o\,;W,\; C,L,E\,\rangle_{A,\mathcal{R}} \;\rightarrow\; \mathsf{Refused}(\mathsf{letter}(o),\,A)}
$$

**The third premise is load-bearing and was missing from the draft.** Without it, at a step with
`C(n) = v` and `letter ∉ A` both this rule and the unguarded `Step-Replay` apply, and the relation
loses determinism at exactly the crash-resume-under-changed-declaration configuration §4e asks a
model checker to explore.

$\mathsf{Refused}$ is a `CompositionRefused`, a programming error: its task fails on the attempt
that raised it, and a spawned child answers its parent `Failed` rather than being retried.

### 4c. The bracket, and the narrowing ruling

$$
\frac{\;A' \le A \qquad \langle\, W',\; C,L,E\,\rangle_{A',\mathcal{R}} \rightarrow^{*} \langle\, v,\; C',L',E'\,\rangle\;}
     {\langle\, \mathsf{carrying}(A',W')\,;W,\; C,L,E\,\rangle_{A,\mathcal{R}} \;\rightarrow\;
      \langle\, W[v/\bullet],\; C',L',E'\,\rangle}
$$

Structurally `Scoped`'s rule on a different coordinate: `Scoped` rewrites *names* along the walk,
this rewrites the *alphabet*. §6 rules that `@needs` on a function is the only **surface** for it;
the rule survives as the rule for entering a decorated function.

> **Confinement (fresh derivations).** If $C = \varnothing$ and
> $\langle W, \varnothing, L, E\rangle_{A,\mathcal{R}} \rightarrow^{*} \kappa'$, then every letter
> yielded in the derivation matches $A$.

**The premise is required.** Without it the statement is false by §4d: a resumed derivation whose
recorded prefix was made under a wider alphabet replays letters ∉ $A$, unguarded. The premise is
exactly what a respawn boundary supplies (§7).

The narrowing premise $A' \le A$ is what buys the theorem at all; under a union rule
($A \sqcup A'$) a nested declaration adds vocabulary and the root's declaration says nothing about
its subtree. **The choice is not narrow-versus-union as a preference; it is whether you want a
confinement theorem.**

### 4d. The replay exemption, and what it actually forces

`Step-Replay` carries **no** alphabet premise. Its defining property is that it *does not evaluate*
the thunk (no world contact, nothing to authorize), and guarding it would make replay a function
of *present* policy, so a run would replay differently tomorrow.

The consequence, stated at the strength the semantics supports:

> $A$ is not re-derivable from a replay. Therefore **something recorded must determine it.**

That is a **disjunction**, not a mandate for one artifact: a carried alphabet row, or a recorded
resolved-program identity (which §7's `modified` row plus artifact digest already supplies), or,
strongest and the recommended form, the **resolution record** of §5: what the handler actually
resolved, minted by the handler rather than echoed from the author, in the
one-mint-many-interpreters discipline. A record that echoes the declaration proves
nothing; a record of the resolution is what makes the declaration falsifiable.

Under crash-resume the prefix replays unguarded and the tail runs guarded against the alphabet as
of resume: new policy binds new work. That is the same class as the deferred **workflow version
marker**, which §7's residue check supersedes in shape.

### 4e. Where the obligations go

| tier | obligation |
|---|---|
| **Lean** | $\le$ a preorder, alphabets a lattice; Confinement (§4c) and not-refused (§5) by induction. Beside `Scopes.lean`'s `flatten_injective_on`. |
| **Quint** | what induction cannot reach: is $\mathsf{Refused}$ reachable under crash-resume with a changed declaration? Is the unguarded-prefix / guarded-tail split sound under interleaving? |
| **Parameterized tests** | the correspondence boundary: does Python's matcher decide $\ell \in A$ the way Lean's $\le$ does? A **third** serialization boundary, after the two the formal estate already names, and precisely where a `startswith` matcher would pass Lean and fail reality. |

**Do not assign the Lean obligation until §4b and §4c are repaired in whatever document carries them**:
a proof effort spent on a false statement is worse than none.

## 5. $\Pi$: the provided repertoire, and its boundary with authorization

The static judgment $\Pi \vdash W : A$ needs a provision side, and **Effective has none**: the
handler stack is closures, `serve` composes middleware that *does* rather than provides,
and `type ToolRunner = Callable[[CallTool[Any]], Any]` (`cost.py:173`) is a bare callable.

**The scope is small, because almost every arm is total.** The ledger writer accepts any
`event_id`; `StoreArtifact` accepts anything; any name can be awaited; `AskLLM` carries no model
field, so the model caller resolves no name. **Only `CallTool(name=…)` is genuinely partial.**

> $\Pi$ is the deployed **tool set**. A layer *wraps*; it does not *provide*.

Two caveats belong with it. **`\mathcal{R}(t)` at `Task-Start` is a second partial resolution**: a
spawn or respawn to an unregistered task name fails a lookup. Although it is outside the
letter space, a "nothing is missing at deploy" check should cover it, or the first
missing-provision failure in production is a respawn to a renamed task. And `route`'s label→handler
map is partial on the author side, where a miss is a crash rather than a `Refused`.

> **Theorem shape:** if $\Pi \vdash W : A$ and $A \le \mathsf{provides}(\Pi)$, then the run cannot
> be refused for a capability reason: *the lint replaces the runtime check.*

**Provision is not authorization.** $\Pi$ answers *"does this exist?"*; the permission cascade
answers *"may you, right now, with these arguments?"* Merging them is the failure mode, and it has
a name here: **an alphabet that encodes permission is a denylist wearing a new hat.**

## 6. The rulings, and the first build step

**`@needs` is the only introduction form.** Every extent in Effective is already a callable
(`route`'s handler map, `gather`'s branch sequence, `recurse`'s leaf/combine, `Scoped.body`), so a
region worth narrowing always has a function to decorate, and a second surface would serve no case
of its own. The honest cost: narrowing a *sub-region of one body* forces an extraction. Carrying
the requirement in the return type is rejected: `type Effect[T] = Generator[WorkflowOp, Any, T]`
(`api.py:33`), so an `R` channel would be phantom, with nothing for `ty` to infer from and no
propagation through `yield from`. **A property of the body is derived from the body or declared
beside it, never smuggled into its type.**

**`Refused` is distinguishable from `Parked`, for free**: an alphabet refusal is a pure function of
the op and its ctx, and subclassing `CompositionRefused` makes it `Unretryable`, so on either engine
its task fails on the attempt that raised it and a spawned child answers its parent `Failed`. A
runtime refusal that escapes a workflow outside `run_child` is still retried to its limit, and
widening that relay is open work.

**The residue check needs no replay/live flag.** The property is one condition, *at the first
thunk execution, $\operatorname{dom}(C)$ must already be exhausted*, which catches an inserted
yield at the frontier and a removed one at the end. It needs no new query: the SDK materializes the
whole checkpoint domain into `ctx._checkpoint_cache` before the first step. It must **not** reach
through the ctx wrappers: their hand-maintained `isinstance` tuple warns that *forgetting* to extend
it when a wrapper is added is the recurring bug (`handlers/absurd.py:336-342`), an obligation a
second consumer inherits. **And the property is unsound as stated under a concurrent `gather`**:
branches interleave, so one branch's first live thunk can precede another's replayed tail. Rule the
frame scope and enrol it in [`tests/_conformance.py`](../tests/_conformance.py) over both engines before building it.

### 6a. `Tool`: the first build step, and it pays for itself

```python
@dataclass(frozen=True)
class Tool[A, R]:
    name: str
    args: type[A]
    result: type[R]


@dataclass(frozen=True)
class Provide[A, R]:
    tool: Tool[A, R]
    impl: Callable[[A], R]
```

One object on both sides of the turnstile: referenced by `@needs` (requirement), registered by a
deployment (provision). The implementation is deliberately not on the `Tool`: that is the binding,
and it belongs to the deployment.

**No `key` property, and NOT because it was forgotten.** An earlier draft gave `Tool` a `key`
returning `compose_key(t"tool:{Segment(self.name)}")`. That is **not the letter**: `step_key` wraps every `Step`
in its own arm (`handlers/base.py:239`), so a `call_tool`'s letter is `step;tool:{name}`. Worse, the
substrate **decouples the dispatch name from the durable key on purpose**: `activate_skill` keys
under `skill:{name},activate` while its `CallTool` name is `skill-disclose` (`skills.py:142-146`),
and `tool_interrupt` mints a *per-call* letter `tool:interrupt,{phase}` whose arity is its
only collision defence. A `frozenset` of name-derived keys would match **zero** of those.

### 6b. The membership relation is UNOWNED: settle it before any matcher ships

$\Pi \vdash W : A$ needs $\ell \in A$ to *mean* something, and nothing here says what. Three facts
make it a decision rather than a detail: a `Step`'s letter carries the `step;` arm, so a declaration
written in tool-names is one arm short; several substrate tools mint **per-call** letters with
trailing coordinates, so membership cannot be set-equality over full keys; it is matching over a
**term prefix with wildcards on trailing coordinates**, which is the structural matcher §4b needs
anyway; and name and key are different axes on purpose, so a `Tool` cannot derive its own letter.

The candidate carries two limits that are part of the decision, not footnotes. The pattern's
vocabulary is the **key namespace, not the dispatch name**: the disclose tool's letters live
under `step;skill:…` while its name is `skill-disclose`, which appears in no letter it mints, so
every pattern is hand-authored per family: a name↔pattern mapping someone must keep true, the
nominal-where-structural shape again. And trailing-only wildcards cannot pin a later coordinate
while wildcarding an earlier one, so the skill family is reachable only at whole-namespace grain,
never per-variant.

**Until this is settled, build `Tool` for the typing win only** (name ↔ args ↔ result bound once)
and leave keys alone. That half is independently valuable and is where the certain value was.

**It is worth building even if the alphabet never ships.** `call_tool[T](name: str, args:
dict[str, Any], schema: type[T])` binds nothing: the substrate already carries **five** reserved
tool-name constants as bare `str`. Four construct `CallTool`s, most with schemas restated by hand
at each construction site: `DISCLOSE_TOOL` (`result_schema=Pin` twice in `skills.py` alone),
`SPAWN_TOOL` (`result_schema=Spawned`, minted once by `SpawnArgs.call`), `INTERRUPT_TOOL` (`compose.py:102`), and `EXECUTE_TOOL`
(`code.py:113`, schema at `:283-293`). The fifth, `CODE_TOOL`, names a tool but constructs no
`CallTool` itself: it is decide-side vocabulary, so its `Tool` binds a name and nothing more. That is
the complaint of *"bare op-key strings that no checker binds to a definition"*,
sitting at the tool seam in our own code, where the failure currently surfaces as
`"unknown tool: {name}"`, an observation the **model** meets mid-turn after the turn is paid for.

**Additive by construction:** `CallTool` is untouched, so the op, `op_key`, the checkpoint value,
the handler arms and replay stay byte-identical, and `call_tool(name=…)` keeps working.

**Open before `runner()` ships:** it replaces `make_tool_runner`'s permissive
unknown-tool-as-observation with a refusal, and the ReAct loop *relies* on routing around unknown
tools (`UnusableToolName` exists for that). A model-invented name and a declared requirement $\Pi$
cannot satisfy are two different failure modes sharing one code path.

## 7. Self-modification: restart at a named point

Nothing prevents it. `SqliteApp.register_task` is a dict assignment; `Respawn` already names the
boundary and already ledgers it.

**Soundness is prefix agreement, not program equality.** $W \approx_C W'$: the two programs consume
the same *letters* over $\operatorname{dom}(C)$ under replay-only rules. Past the frontier they may
differ arbitrarily.

**At a generation boundary the condition is vacuous.** Respawn cuts to $C = \varnothing$ (*"the next
generation replays no history"*), and $\approx_\varnothing$ relates **all** programs, so every
generation boundary is an unconditionally safe modification point. The property Cordis needs
quiescence machinery for is a corollary of emptiness here.

**Zero new control ops:** `StoreArtifact` (program bytes, content-addressed) + `AppendLedgerRow`
(the fact plus digest) + `Respawn` (adoption). The registry mutation is world state recorded by
digest, not tape.

**The privilege ratchet, with its limit.** Carrying the alphabet forward under §4c's narrowing rule
means a chain may rewrite its own code arbitrarily and still never call anything an earlier
generation could not. **Limit, per §3a:** the ratchet governs what generations *call*, not the door
itself, and $A' \le A$ at the boundary is asserted rather than enforced.

**The residue check supersedes the version marker in shape.** A marker is a nominal proxy someone
must remember to bump, which is a recurring defect class; the structural property is
$\operatorname{dom}(C)$ exactly consumed at the frontier. Owed **independently of this axis**: every
production redeploy is already a mid-task code change, since Absurd is pull-only and in-flight
tasks resume by replay on new code.

## 8. The operator at the boundary

The outer-loop operator and a human steering are **one role at different binding times**: the
ratified authored / mission-profile / just-in-time table already contains the human as its third
row.

**A capability change can only happen at a derivation boundary**, because $A$ and $\mathcal{R}$ are
context (§4a). This is a consequence of the ratified semantics, not a discovered impossibility;
the semantics is a design choice, and a durable note must not let a definition impersonate a
theorem. The rule already exists: `Self-Modify` concludes a terminal configuration whose successor
begins at $A',\mathcal{R}'$.

**The human is already at that boundary.** On budget exhaustion a chain *"parks durably and a human
answers with `add_generations` or a `stop`"*. So the boundary is already safe, already recorded (the
`respawned` ledger row), and already governed. Adding a repertoire to that answer is one more field.

**An outsider cannot end a task, and must not.** An externally forced kill is not a deterministic
transition, so the loop **polls**: the known-door invariant. One recorded poll, two outcomes:

| response | effect | cost |
|---|---|---|
| **redirect** | continue this derivation, splice a user message | cheap, no boundary |
| **repertoire / grant** | `Again(state)` → boundary → new $A, \Pi$ | a generation, plus the prologue |

**The cost, and it is structural.** `respawn` re-enters the whole workflow, so *"code textually
before this call runs live and un-checkpointed in EVERY generation… a `call_tool` up there sends
mail once per generation."*

> $C = \varnothing$ is **one fact wearing two faces**: it is why the boundary is unconditionally
> safe and why the prologue re-executes. **Safety comes from forgetting, and forgetting is what
> costs.** No version of this boundary is both cheap and universally safe.

So frequent steering and cheap steering pull against each other, and the operator's grain **is** the
generation grain: `respawn` runs `step` once per generation, so the equivalence with a per-turn
interrupt is exact only when a generation is a turn. Latency is bounded by the worker's run window, not by
the durable-sleep wake gap: a respawn's successor is immediately runnable, so a chain continues
freely *within* a window and stalls only at its edge.

**What the tape shows today, and what it does not.** The interrupt poll *is* a recorded op, so an
interruptible run's tape differs from an uninterruptible one's. The governance gap is not the
trajectory bytes but the **closure identity**: which channel the poll reads, and who may write to
it, is unrecorded. Provenance stamping is the requirement: an operator's tier must be a recorded
field, never inferred, or an optimizer's push and a human's authoritative overrule are
indistinguishable on the tape.

## 9. Build order and triggers

| step | depends on | why now, or not |
|---|---|---|
| **`Tool` / `Provide` objects** | nothing | pays for itself on the typing win alone (§6a); the five bare constants are the proof (§6b) |
| **The `carried` resolution record** | nothing | §4d's disjunction; needs no $\Pi$ |
| **The residue check** | nothing | owed independently: every redeploy is a mid-task code change |
| **`op_key`'s `Respawn` arm** | §3a's ruling | makes the substrate's line match the stated one; mind the `placed_key` subtlety |
| **`runner()` / $\Pi$** | the unknown-tool reconciliation (§6a) | a behaviour change to the ReAct loop |
| **`@needs`** | $\Pi$ | duals: both or neither |
| **Peek / Emit rules** | a decision on §4a | only if $E$-as-state stays load-bearing |
| **Lean / Quint obligations** | §4b and §4c repaired | do not assign a proof on a false statement |

**Open, and measurable rather than arguable:** the generation grain (one turn vs K), and the
prologue tax against interrupt frequency. Both want measurement before commitment.

**Open, and a genuine fork:** whether a pure redirect should *also* be a boundary: uniformity (one
outcome, simplest semantics) against cost (the prologue tax, paid per boundary).
