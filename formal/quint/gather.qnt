// Effective — gather + totality (the suspend-in-gather bug), in Quint
//
// v1 of the Quint model (v0 = `effective.qnt`). This is where the model checker earns
// its keep: it models `gather` — branches running concurrently under their own
// continuations, the W-interleaving Lean's Step relation punted on — and checks
// **I2 totality** faithfully.
//
// I2: every reachable non-terminal configuration has, for each
// active op-head, a *defined transition OR a defined park*. The bug modeled: an
// `Await` inside a gather branch had NEITHER on the durable path — a hole in the
// relation, which surfaced as a silent suspend-in-gather crash. We encode I2 directly
// as `total` (every reachable head has a rule), so the bug is "an in-branch await the
// handler has no rule for", found as a reachable config where `total` fails.
//
// NOTE (I2 vs I3): an await waiting on an *absent* event is legitimately **Parked** —
// a defined behavior, so I2-OK. Whether that event ever arrives (so the run can
// finish) is *liveness* (I3), a separate, temporal property — a later rung, not this
// one. `total` deliberately does not conflate the two.

module gather {
  type Op =
    | Step(int)
    | Await(int)
    | Ledger(int)
    | Gather        // spawns `branchSpecs` (v1: a single, non-nested gather)

  // --- state ---
  var main: List[Op]            // the top-level workflow continuation
  var branches: List[List[Op]]  // active gather branches, each its own continuation
  var chk: Set[int]             // recorded checkpoint names
  var led: List[int]            // append-only ledger
  var evt: Set[int]             // delivered events

  pure val events: Set[int] = Set(7, 8)

  // The gather's branches. Branch 0 contains an `Await(7)` — the suspend-in-gather
  // trigger; branch 1 is await-free.
  pure val branchSpecs: List[List[Op]] =
    [ [Step(2), Await(7), Ledger(20)],
      [Step(3), Ledger(21)] ]

  // Top-level workflow: a step, then the gather, then a final ledger commit.
  pure val workflow: List[Op] = [Step(1), Gather, Ledger(30)]

  // HANDLER TOGGLE (the rung-1 experiment): the *fixed* handler defines a rule for an
  // await inside a branch (suspend → resume when the event arrives). The *buggy*
  // handler has none: a hole at await-in-branch. Flip to false and
  // `verify --invariant=total` finds a reachable config the handler can't handle.
  pure val handlerFixed: bool = true

  // === Imported from Lean as explicit assumptions (the §8.3 division of labor) ===
  //
  // Quint EXPLORES reachability/interleavings; it does NOT re-derive the pure facts
  // Lean proves. It *imports* them as assumptions so the model stays small. Each is
  // discharged in `formal/lean/` (axiom footprint shown); were any false, this model's
  // results would be vacuous. Naming them here is the point — the reliance is explicit
  // and auditable, not silently embodied.

  // T2 — determinism. Lean: `Effective.Step.step_deterministic` ([propext]). Given the
  // recorded results, the transition relation is a partial function (one rule fires,
  // outcome unique). RELIED ON: each op has a single deterministic clause and the
  // guards are mutually exclusive, so we never model nondeterministic rule-racing.
  // Step-Replay being a no-op (L2.5) is witnessed at the model level by `chk.union`
  // being idempotent — re-recording an already-present name changes nothing.
  assume T2_determinism = true

  // T3 — append-only + idempotent. Lean: `Effective.Ledger.ledger_monotone` ([propext])
  // and `ledger_idem` (axiom-free). The ledger only grows (prefix order) and never
  // duplicates an eid. RELIED ON: `led` is only ever extended by a guarded append; the
  // `ledgerNodup` invariant below is the model-side reflection (Lean owns the general
  // over-runs proof).
  assume T3_appendOnly = true

  // L1.5 — gather naming injectivity. Lean: `Effective.Keys.gather_naming_injective`
  // ([propext]). Across a gather, (branch index, inner key) ↦ name is injective; branch
  // namespaces are disjoint. RELIED ON: checkpoint names / eids are a flat int namespace
  // and distinct logical checkpoints are given distinct ints (branch 0 uses 2, branch 1
  // uses 3, …). That distinctness STANDS FOR what L1.5 guarantees — were two branches to
  // reuse an int here, it would model a name collision L1.5 rules out. Importing it (vs
  // re-deriving) IS §8.3: Lean owns injectivity so Quint needn't carry prefixed keys and
  // re-check disjointness at every state. (Parent↔branch disjointness additionally needs
  // generator freshness — the runtime's obligation, not a Lean theorem.)
  assume L15_gatherNamingInjective = true

  pure def ledHas(l: List[int], x: int): bool =
    l.indices().exists(i => l[i] == x)

  // The handler's rule for an op head, by where the head stands. The actions fire on it and
  // the invariants read it, so a rule taken out of the handler is a hole `total` sees.
  type Rule = Advance | ResumeOn(int) | NoRule

  pure def rule(op: Op, inBranch: bool): Rule =
    match op {
      | Await(e) => if (inBranch and not(handlerFixed)) NoRule else ResumeOn(e)
      | Gather   => if (inBranch) NoRule else Advance   // a nested gather is not modeled
      | _        => Advance
    }

  pure def fires(r: Rule, delivered: Set[int]): bool =
    match r {
      | Advance     => true
      | ResumeOn(e) => delivered.contains(e)
      | NoRule      => false
    }

  pure def hasRule(r: Rule): bool =
    match r {
      | NoRule => false
      | _      => true
    }

  // A head parked on an event the environment has yet to deliver.
  pure def deliveryUnblocks(r: Rule, delivered: Set[int]): bool =
    match r {
      | ResumeOn(e) => events.contains(e) and not(delivered.contains(e))
      | _           => false
    }

  action init = all {
    main' = workflow,
    branches' = [],
    chk' = Set(),
    led' = [],
    evt' = Set(),
  }

  // step the top-level workflow head. Self-guarded (safe to query enabledness under
  // fairness) AND gated on the gather *join*: while a gather is in progress
  // (`branches` non-empty) main is blocked — the (Gather) rule's continuation runs only
  // after all branches join (§2.4). The `Gather` op itself fires when `branches == []`,
  // spawns the branches, and thereby blocks main until they drain + `joinGather` clears.
  action stepMain = if (main.length() == 0 or branches.length() > 0)
    all { false, main' = main, branches' = branches, chk' = chk, led' = led, evt' = evt }
  else {
    val op = main.head()
    val rest = main.tail()
    all {
      fires(rule(op, false), evt),
      match op {
        | Step(n) => all {
            chk' = chk.union(Set(n)), main' = rest,
            branches' = branches, led' = led, evt' = evt }
        | Ledger(eid) => all {
            led' = if (not(ledHas(led, eid))) led.append(eid) else led, main' = rest,
            branches' = branches, chk' = chk, evt' = evt }
        | Await(_) => all {                       // resumes once its event is delivered
            main' = rest,
            branches' = branches, chk' = chk, led' = led, evt' = evt }
        | Gather => all {                         // spawn the branches
            branches' = branchSpecs, main' = rest,
            chk' = chk, led' = led, evt' = evt }
      }
    }
  }

  // branch indices that still have work
  val activeBranches: Set[int] =
    branches.indices().filter(i => branches[i].length() > 0)

  // step one active branch head (the interleaving Quint exists to explore).
  // NOTE: branch ledger rows append in *completion* order, not §2.4's branch-index
  // order — interleaving is exactly what this model explores; the index-ordered
  // concatenation (and the replay-stability that rests on it) is Lean's via T2, not
  // re-imposed here.
  action stepBranch = if (activeBranches == Set())
    all { false, main' = main, branches' = branches, chk' = chk, led' = led, evt' = evt }
  else {
    nondet i = oneOf(activeBranches)
    val b = branches[i]
    val op = b.head()
    val rest = b.tail()
    all {
      fires(rule(op, true), evt),               // the buggy handler has no rule for an await here
      match op {
        | Step(n) => all {
            chk' = chk.union(Set(n)), branches' = branches.replaceAt(i, rest),
            main' = main, led' = led, evt' = evt }
        | Ledger(eid) => all {
            led' = if (not(ledHas(led, eid))) led.append(eid) else led,
            branches' = branches.replaceAt(i, rest),
            main' = main, chk' = chk, evt' = evt }
        | Await(_) => all {
            branches' = branches.replaceAt(i, rest),
            main' = main, chk' = chk, led' = led, evt' = evt }
        | Gather => all {                         // never fires: a nested gather has no rule
            branches' = branches, main' = main, chk' = chk, led' = led, evt' = evt }
      }
    }
  }

  // the gather join: once every branch has drained (all inner programs empty), collapse
  // `branches` back to empty — this completes the join and unblocks `stepMain`. (We
  // collapse the drained `[[], []]` to `[]`; the gathered checkpoint/ledger effects are
  // already merged into the shared state, in completion order — see the stepBranch note.)
  action joinGather = all {
    branches.length() > 0,
    activeBranches == Set(),
    branches' = [],
    main' = main, chk' = chk, led' = led, evt' = evt }

  // the environment delivers a specific event (always enabled — models the outside
  // world; parameterized so fairness can target the awaited event)
  action deliverEvent(e: int): bool = all {
    evt' = evt.union(Set(e)),
    main' = main, branches' = branches, chk' = chk, led' = led }

  action deliver = {
    nondet e = oneOf(events)
    deliverEvent(e)
  }

  action step = any { stepMain, stepBranch, joinGather, deliver }

  // --- invariants ---

  // I2 (totality): every reachable configuration's active heads, the main head and every
  // branch head, have a rule to fire on, a step or a resume. The buggy handler leaves the
  // in-branch Await(7) with none, and a nested gather in `branchSpecs` would have none.
  val total: bool =
    (main.length() == 0 or hasRule(rule(main.head(), false))) and
    branches.indices().forall(i =>
      branches[i].length() == 0 or hasRule(rule(branches[i].head(), true)))

  // An await's effect waits for its event: branch 0's row after `Await(7)` lands only once
  // event 7 is delivered. The stuckness checks cannot see a rule that fires too early.
  val awaitHonored: bool = ledHas(led, 20) implies evt.contains(7)

  // T3 cross-check carried over: the ledger never holds a duplicate.
  val ledgerNodup: bool =
    led.length() == led.foldl(Set(), (acc, x) => acc.union(Set(x))).size()

  // === I3 liveness (temporal — a new rung) ===
  //
  // I2 (totality) said the handler always *can* act. I3 says it always eventually
  // *makes progress to rest*: the whole workflow eventually terminates. This is a
  // temporal property, and it is **conditional on fairness** — without it the trivial
  // counterexample is "the environment delivers other events forever and never the
  // awaited one," or "the scheduler stutters and never steps the parked branch." So
  // the honest statement is `fairness implies eventuallyTerminates`.
  //
  // CHECK THESE WITH `--backend=tlc` (TLC fully supports temporal+fairness; Apalache's
  // temporal support is experimental and prompts interactively). E.g.
  //   quint verify gather.qnt --temporal=liveness --max-steps=14 --backend=tlc

  val terminated: bool = main.length() == 0 and activeBranches == Set()

  // --- deadlock-freedom (a whole-STATE safety invariant, sibling to the per-head `total`) ---
  // `total` checks every reachable op-head has a rule. `noDeadlock` checks every reachable
  // non-terminal *configuration* can make progress now, or has a pending await that a
  // delivery would unblock AND the handler can resume. It catches a stuck *state* even when
  // it isn't a missing-rule one, which `total` cannot see. Quint invariants are state
  // predicates rather than `enabled(action)`, so the heads' enabledness reads `rule` and
  // `fires`, as the actions do, and the join and the main gate restate their guards.
  val mainStepEnabled: bool =
    main.length() > 0 and branches.length() == 0 and fires(rule(main.head(), false), evt)

  val branchStepEnabled: bool =
    activeBranches.exists(i => fires(rule(branches[i].head(), true), evt))

  val joinEnabled: bool = branches.length() > 0 and activeBranches == Set()

  // a not-yet-delivered, deliverable event whose await the handler will resume on delivery
  val deliverWouldHelp: bool =
    (main.length() > 0 and branches.length() == 0 and
       deliveryUnblocks(rule(main.head(), false), evt))
    or activeBranches.exists(i => deliveryUnblocks(rule(branches[i].head(), true), evt))

  // I2-adjacent: no reachable configuration is genuinely stuck. FIXED handler: holds.
  // BUGGY handler: fires at `[[Await(7),Ledger(20)],[]]`, the in-branch await with no rule,
  // which neither steps nor waits on a delivery.
  val noDeadlock: bool =
    terminated or mainStepEnabled or branchStepEnabled or joinEnabled or deliverWouldHelp

  // Expected to be violated: a head parks on an undelivered event, so `awaitHonored` is checked
  // where it can fail.
  val neverParked: bool = not(deliverWouldHelp)

  val vars = (main, branches, chk, led, evt)

  temporal eventuallyTerminates = eventually(terminated)

  // Weak fairness: an action continuously enabled eventually fires. We need the
  // awaited event delivered (deliverEvent(7) is always enabled) and the work stepped.
  // NOTE: aggregate `weakFair(stepBranch)` over the nondet branch pick is sufficient
  // here ONLY because branches are finite — once branch 1 drains, branch 0 is the sole
  // active one and WF must step it. Nested/looping branches (the 'Next' item) would
  // need per-branch or strong fairness to avoid starvation; aggregate WF would not do.
  temporal fairness =
    weakFair(deliverEvent(7), vars) and
    weakFair(stepMain, vars) and
    weakFair(stepBranch, vars) and
    weakFair(joinGather, vars)

  // I3: under fairness, the workflow terminates. FIXED handler: holds. BUGGY handler:
  // fails — the in-branch await never resumes (its step is never enabled, so fairness
  // cannot force it), so the gather never completes. Liveness pins the buggy handler's
  // permanent stall, a failure mode distinct from (though here caused by) the I2 hole.
  temporal liveness = fairness implies eventuallyTerminates

  // NON-VACUITY GUARD. `liveness` is vacuously true if `fairness` is satisfiable by no
  // behavior. This property asserts "no fair behavior ever terminates" — so its
  // **EXPECTED RESULT IS A VIOLATION**: the counterexample TLC returns is a fair,
  // terminating run (routing through the Await(7) resume), which proves `fairness` is
  // satisfiable and `liveness` is therefore not hollow. Check (expect a violation):
  //   quint verify gather.qnt --temporal=nonVacuityWitness --max-steps=14 --backend=tlc
  temporal nonVacuityWitness = fairness implies always(not(terminated))
}
