// Effective — budget accrual under gather: confluence, operationally (v2 of the Quint models)
//
// The companion to `formal/lean/Effective/Budget.lean`. Lean owns the ALGEBRA: it proves
// Model-B accrual is permutation-invariant for ALL inputs (`executedB_perm_invariant`) by
// modeling the concurrent branches as a `Multiset` — confluence definitional, the N!
// orderings collapsed to one theorem. It also exhibits the Model-S counterexample as a
// decidable fact (`shared_gate_order_dependent`) and the "lock is not enough" coincidence
// (`shared_total_coincides_but_trip_differs`, axiom-free).
//
// This model is the §8.3 OPERATIONAL cross-check: that the divergence is reachable in the
// step relation under NONDETERMINISTIC scheduling — a scheduler actually produces it — not a
// re-proof of the algebra. Same split as gather.qnt: Quint explores interleavings; Lean
// proves the pure facts. gather.qnt:126-130 punted exactly this (branch effects in
// completion order, index-order confluence "Lean's via T2"); here it is made explicit.

module budget_confluence {
  type Op = { id: int, cost: int }

  // --- state ---
  var branches: List[List[Op]]   // remaining ops per branch; interleaving is what we explore
  var spent: int                 // Model S: one shared meter
  var spentPer: List[int]        // Model B: a meter per branch
  var executed: Set[int]         // the observable: op ids that ran

  pure val limit: int = 10       // the shared ceiling (Model S)
  pure val perBudget: int = 5    // limit / #branches — the per-branch partition (Model B)

  // The scenario: branch 0 = [a1(6), a2(6)], branch 1 = [b1(6)].
  pure val branchSpecs: List[List[Op]] =
    [ [ { id: 10, cost: 6 }, { id: 11, cost: 6 } ],
      [ { id: 20, cost: 6 } ] ]

  // The schedule-INDEPENDENT outcome (each branch bounded by its own recorded prefix):
  // branch 0 runs a1 then a2 refuses (spentPer 6 >= 5); branch 1 runs b1. Computable from
  // inputs — this is the confluent semantics Lean proves invariant.
  pure val canonicalExecuted: Set[int] = Set(10, 20)

  pure val noOps: List[Op] = []

  // TOGGLE (the gather.qnt `handlerFixed` idiom): the accrual model under test.
  //  false = Model B: each branch gates against its OWN budget — confluent. The gate below
  //          HOLDS (Apalache confirms operationally what Lean proves for all inputs).
  //  true  = Model S: one shared meter, gated mid-fold. Flip and `verify --invariant=confluent`
  //          returns the [a1,a2,b1] schedule with executed {10,11} != {10,20} — the operational
  //          witness of Effective.Budget.shared_gate_order_dependent. NOTE the model is atomic
  //          (stepBranch reads-then-writes `spent` in one action = a LOCKED meter) and STILL
  //          non-confluent: the lock fixes the total, not the trip.
  pure val sharedMeter: bool = false

  // === §8.3 imported from Lean (the pure facts this model does NOT re-derive) ===
  // Effective.Budget.operational_run_eq_executedB: the interleaved per-branch fold reaches the
  // multiset aggregate `executedB` on EVERY draining schedule (proved by a conservation law,
  // no interleaving enumeration) — so Model-B confluence holds for all inputs. This Quint run
  // is the operational cross-check on the witness scenario, not the source of that ∀.
  assume Lean_operationalRunEqAggregate = true

  action init = all {
    branches' = branchSpecs, spent' = 0, spentPer' = [0, 0], executed' = Set() }

  // branch indices with work left
  val activeBranches: Set[int] =
    branches.indices().filter(i => branches[i].length() > 0)

  // step one active branch head — the interleaving Quint exists to explore. The gate refuses
  // an op once the relevant budget is exhausted (spend is monotone, so refusal halts the
  // branch: every later op refuses too).
  action stepBranch = if (activeBranches == Set())
    all { false, branches' = branches, spent' = spent, spentPer' = spentPer, executed' = executed }
  else {
    nondet i = oneOf(activeBranches)
    val op = branches[i].head()
    val exhausted = if (sharedMeter) spent >= limit else spentPer[i] >= perBudget
    if (exhausted)
      all {                                    // PARK: refuse this op; the branch halts
        branches' = branches.replaceAt(i, noOps),
        spent' = spent, spentPer' = spentPer, executed' = executed }
    else
      all {                                    // RUN: accrue into the relevant meter, advance
        executed' = executed.union(Set(op.id)),
        spent' = if (sharedMeter) spent + op.cost else spent,
        spentPer' = if (sharedMeter) spentPer else spentPer.replaceAt(i, spentPer[i] + op.cost),
        branches' = branches.replaceAt(i, branches[i].tail()) }
  }

  val atTerminal: bool = activeBranches == Set()

  // Stutter at a drained configuration so a terminal state is not reported as a deadlock
  // (the gather.qnt role its always-enabled `deliver` plays). Safety-only; enabled solely
  // once all branches have drained or parked.
  action stutter = all {
    atTerminal,
    branches' = branches, spent' = spent, spentPer' = spentPer, executed' = executed }

  action step = any { stepBranch, stutter }

  // --- invariant ---

  // Confluence: every reachable terminal's executed bag is the schedule-independent value.
  //  Model B: HOLDS on every interleaving.
  //  Model S: VIOLATED — a reachable terminal has executed {10,11} (a1,a2 ran, b1 refused),
  //           differing from another schedule's {10,20}. The counterexample IS the schedule.
  val confluent: bool = atTerminal implies (executed == canonicalExecuted)

  // Expected to be violated: a schedule drains every branch, so `confluent` is checked at a
  // terminal it can fail on.
  val neverAtTerminal: bool = not(atTerminal)

  val vars = (branches, spent, spentPer, executed)
}
