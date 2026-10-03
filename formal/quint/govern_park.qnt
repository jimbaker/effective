// Effective: the `govern` park protocol, park -> request -> deliver -> resume -> worker-death
//
// The companion to `formal/lean/Effective/Govern.lean`. Lean owns the RULING (the conjunctive
// fold: refuse dominates park, order-freedom, no silent proceed) and states that it cannot see
// the protocol around it. This model is that protocol, the reachability half that belongs to
// Quint.
//
// ONE model covers BOTH gates: budget's grant and permission's approval differ only in what the
// answer MEANS, and the answer's meaning is the policy's business (`Resolution.answers[policy]`).
// What is modeled here is the control shape they share.
//
// The mechanism under test: a park is a suspension resolved by replay, never a captured
// continuation. The worker suspends, and on resume the computation is RE-RUN, re-deriving the
// same deterministically-named await from recorded state and re-binding it by name. The safety
// argument therefore rests on one claim: **the park name is a function of durable state alone**.
// `govern`'s name is `govern:{gate}:{run_id}:{pass_n}` where `pass_n` is the number of
// resolutions already absorbed, so it is. `durableName = false` flips that to an in-memory
// counter and the checker finds the stranded grant.
//
// === imported from Lean (facts this model does NOT re-derive) ===
// Effective.Govern.combine: the per-pass ruling is a total, deterministic function of the
// policies' verdicts, with refuse dominating park. Modeled here as an atomic `evaluate` action
// whose branch structure mirrors it: the ruling is Lean's, the interleaving is Quint's.
// Effective.EnforceMeasured.run_overshoot_bound: spend stays within ceiling + one call.

module govern_park {
  // The op needs this much granted headroom before any policy will let it through. (Read it as
  // dollars for the budget gate, or as "one approval" for the permission gate with need=1.)
  pure val need: int = 10
  pure val grantSize: int = 4      // what one delivered resolution is worth
  pure val maxPasses: int = 3      // govern's DEFAULT_MAX_PASSES, scaled down for checking

  // TOGGLE (the gather.qnt `handlerFixed` idiom): where the park name comes from.
  //  true  = the shipped design: pass index = |absorbed|, a function of DURABLE state, so a
  //          fresh worker re-derives the same name. All four invariants hold.
  //  false = the tempting bug: an in-process counter. It survives a clean run and breaks only
  //          across worker death — `verify --invariant=awaitingMatchesPass` returns the trace
  //          where a revived worker awaits a name whose answer was already consumed.
  pure val durableName: bool = true

  // --- durable state (survives worker death — the ledger side of two bookkeepers) ---
  var absorbed: List[int]      // pass indices whose resolution has been folded in, in order
  var deliveredSet: Set[int]   // pass indices a grantor has answered (Absurd events, by name)
  var requested: Set[int]      // pass indices for which a park request was emitted

  // --- ephemeral state (lost on worker death — the checkpoint/execution side) ---
  var alive: bool
  var pending: bool            // this worker is parked, awaiting a resolution
  var awaiting: int            // the pass index THIS worker actually awaits (its park's name)
  var counter: int             // the BUGGY name source: an in-process pass counter

  // --- observable ---
  var outcome: str             // "running" | "proceeded" | "refused"

  // The pass index the shipped gate awaits on: purely a function of durable state.
  val passN: int = absorbed.length()

  // The index this worker actually emits a request for — the toggle's whole content.
  val requestIndex: int = if (durableName) passN else counter

  // The meter, RE-DERIVED from the durable record rather than carried in memory. This is the
  // two-bookkeepers invariant in miniature: nothing a crash can lose feeds the gate's decision.
  val granted: int = absorbed.length() * grantSize

  action init = all {
    absorbed' = [], deliveredSet' = Set(), requested' = Set(),
    alive' = true, pending' = false, awaiting' = 0, counter' = 0, outcome' = "running" }

  // The gate evaluates one pass: the ruling's branch structure (Lean's `combine`), realized.
  action evaluate = all {
    alive, outcome == "running", not(pending),
    if (granted >= need)
      all { outcome' = "proceeded", requested' = requested, pending' = false,
            awaiting' = awaiting, counter' = counter }
    else if (passN >= maxPasses)
      all { outcome' = "refused", requested' = requested, pending' = false,
            awaiting' = awaiting, counter' = counter }
    else
      all { outcome' = "running",                     // PARK: emit exactly one request, by name
            requested' = requested.union(Set(requestIndex)),
            pending' = true,
            awaiting' = requestIndex,                 // ...and await THAT name
            counter' = counter + 1 },
    absorbed' = absorbed, deliveredSet' = deliveredSet, alive' = alive }

  // A grantor answers some outstanding request. Deliveries are by NAME and idempotent (a Set),
  // which is what makes a re-delivery after a crash harmless.
  action deliver = all {
    requested.exclude(deliveredSet) != Set(),
    nondet i = oneOf(requested.exclude(deliveredSet))
    deliveredSet' = deliveredSet.union(Set(i)),
    absorbed' = absorbed, requested' = requested,
    alive' = alive, pending' = pending, awaiting' = awaiting, counter' = counter,
    outcome' = outcome }

  // The parked worker's await fires when an event arrives under THE NAME IT IS AWAITING — the
  // re-bind-by-name mechanism. It then folds the answer into the durable record and advances.
  // Note the asymmetry the bug lives in: the await is satisfied by `awaiting`, but the record
  // advances by `passN`. They coincide only when the name came from durable state.
  action resume = all {
    alive, pending, deliveredSet.contains(awaiting),
    absorbed' = absorbed.append(passN),
    pending' = false,
    deliveredSet' = deliveredSet, requested' = requested, alive' = alive,
    awaiting' = awaiting, counter' = counter, outcome' = outcome }

  // Worker death: every ephemeral bit is lost. Nothing durable moves.
  action die = all {
    alive, outcome == "running",
    alive' = false, pending' = false,
    absorbed' = absorbed, deliveredSet' = deliveredSet, requested' = requested,
    awaiting' = awaiting, counter' = counter, outcome' = outcome }

  // A fresh worker picks the task up and REPLAYS — no captured continuation, no in-flight state.
  // Under the shipped design it re-derives `passN` from `absorbed` and re-awaits the same name;
  // under the bug it restarts its in-process counter, and the name drifts.
  action revive = all {
    not(alive),
    alive' = true, pending' = false, awaiting' = 0, counter' = 0,
    absorbed' = absorbed, deliveredSet' = deliveredSet, requested' = requested,
    outcome' = outcome }

  val atTerminal: bool = outcome != "running"

  // Stutter at a settled gate so a terminal state is not reported as a deadlock (the
  // budget_confluence.qnt / gather.qnt idiom). Safety-only.
  action stutter = all {
    atTerminal,
    absorbed' = absorbed, deliveredSet' = deliveredSet, requested' = requested,
    alive' = alive, pending' = pending, awaiting' = awaiting, counter' = counter,
    outcome' = outcome }

  action step = any { evaluate, deliver, resume, die, revive, stutter }

  // --- invariants ---

  // FAIL-CLOSED UNDER CRASH: the op is never forwarded unless the gate was actually satisfied.
  // A crash between park and answer must not look like an approval. This is the one an operator
  // is really trusting when they leave a run parked overnight.
  val failClosed: bool = (outcome == "proceeded") implies (granted >= need)

  // EXACTLY-ONCE ABSORPTION: each pass's resolution is folded in exactly once, in order — so
  // `absorbed` is precisely [0, 1, ..., n-1]. A re-delivery or a re-run after death cannot
  // double-count a grant (which would silently raise the ceiling).
  val exactlyOnce: bool = absorbed.indices().forall(i => absorbed[i] == i)

  // ONE PARK, ONE NAME: a parked worker awaits exactly the name its durable position implies.
  // This is the no-call/cc claim itself — the whole safety argument is that a fresh worker
  // re-derives the SAME name from recorded state. Flip `durableName` to false and the checker
  // returns the trace where a revived worker awaits a name whose answer was already consumed:
  // the stale-approval bug (`permission.human`'s run-scoping docstring), reachable in three steps after a worker death.
  val awaitingMatchesPass: bool = pending implies (awaiting == passN)

  // BOUNDED: the gate settles within maxPasses resolutions rather than parking forever — the
  // termination guard `govern`'s driver enforces (and the reason it has one).
  val bounded: bool = absorbed.length() <= maxPasses

  // Everything a safety run should check at once.
  val safe: bool = failClosed and exactlyOnce and awaitingMatchesPass and bounded

  // LIVENESS (I3-grade, TLC — the gather.qnt weakFair idiom). Fairness here is: crashes
  // eventually stop (`eventually(always(alive))` — the crash-stop assumption; without it an
  // adversary that kills the worker forever starves any progress, and no protocol can fix that),
  // and every other continuously-enabled action eventually fires. `die` is deliberately NOT
  // fair: a crash is possible, never obligatory.
  temporal fairness =
    eventually(always(alive)) and
    weakFair(evaluate, vars) and
    weakFair(deliver, vars) and
    weakFair(resume, vars) and
    weakFair(revive, vars)

  // I3: under fairness the gate SETTLES — it does not park forever. A parked run that a grantor
  // answers reaches a verdict even across worker deaths; and because `bounded` caps the passes,
  // a grantor who keeps under-granting still terminates the gate rather than looping.
  temporal liveness = fairness implies eventually(atTerminal)

  // NON-VACUITY GUARD (the gather.qnt precedent): `liveness` is hollow if no behavior satisfies
  // `fairness`. This property claims "no fair behavior ever settles", so its EXPECTED RESULT IS
  // A VIOLATION — the counterexample TLC returns is a fair, settling run.
  //   quint verify govern_park.qnt --temporal=nonVacuityWitness --max-steps=10 --backend=tlc
  temporal nonVacuityWitness = fairness implies always(not(atTerminal))

  val vars = (absorbed, deliveredSet, requested, alive, pending, awaiting, counter, outcome)
}
