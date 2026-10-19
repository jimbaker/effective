// Effective: a settlement's apply and its undo at the ERP, across two attempts of one task.
//
// An attempt sends an Apply under the step's key and records the answer as its checkpoint. The
// settlement then decides to keep the match or compensate, and compensating sends an Unapply
// that names the Apply it undoes. The first attempt's lease can expire at any point; a second
// attempt then replays the first's checkpoints and runs the rest. A call already sent cannot be
// recalled, so it lands at the ERP whenever the network delivers it, whichever attempt sent it.
//
// The ERP keeps claims on the Apply's key, each written in the same transaction as the match it
// records: an Apply finding any claim lands nothing, and otherwise claims the key as applied,
// matching unless another key's match already stands; an Unapply finding the key claimed as undone
// lands nothing, and otherwise claims it so and removes the match. The run may start with another
// key's match standing.
//
// Assumed, as the engine's question: an attempt whose lease expired writes no checkpoint. The
// model writes a checkpoint only for the attempt that holds the lease.

module erp_claim {
  pure val ATTEMPTS: Set[int] = Set(1, 2)

  // --- toggle: false is the ruled design ---
  pure val buggyNoClaim: bool = false  // booksAreTheDecision

  // A call in flight: the attempt that sent it, and "apply" or "unapply".
  type Call = (int, str)

  // pc: "start", "applying", "deciding", "unapplying", "done".
  // claims: a subset of "applied" and "undone", on the Apply's key.
  var matched: bool
  var foreign: bool          // the run started with another key's match standing
  var claims: Set[str]
  var inflight: Set[Call]
  var answered: Set[Call]
  var current: int
  var pc: int -> str
  var applied: bool         // checkpoint: the Apply's answer
  var decision: str         // checkpoint: "none", "keep" or "compensate"
  var unapplied: bool       // checkpoint: the Unapply's answer
  var staleLanded: bool     // a call from an attempt without the lease reached the ERP
  var compensated: bool     // an Unapply removed the match

  action init = all {
    nondet standing = oneOf(Set(false, true))
    all { matched' = standing, foreign' = standing },
    claims' = Set(), inflight' = Set(), answered' = Set(), current' = 1,
    pc' = ATTEMPTS.mapBy(_ => "start"), applied' = false, decision' = "none",
    unapplied' = false, staleLanded' = false, compensated' = false,
  }

  action unchangedErp = all {
    matched' = matched, foreign' = foreign, claims' = claims, staleLanded' = staleLanded,
    compensated' = compensated,
  }

  // --- an attempt ---

  action start(a: int): bool = all {
    a == current, pc.get(a) == "start",
    if (applied) all { pc' = pc.set(a, "deciding"), inflight' = inflight }
    else all { pc' = pc.set(a, "applying"), inflight' = inflight.union(Set((a, "apply"))) },
    answered' = answered, current' = current, applied' = applied, decision' = decision,
    unapplied' = unapplied, unchangedErp,
  }

  action heardApply(a: int): bool = all {
    a == current, pc.get(a) == "applying", answered.contains((a, "apply")),
    pc' = pc.set(a, "deciding"), applied' = true,
    inflight' = inflight, answered' = answered, current' = current, decision' = decision,
    unapplied' = unapplied, unchangedErp,
  }

  action decide(a: int): bool = all {
    a == current, pc.get(a) == "deciding",
    nondet chosen = oneOf(Set("keep", "compensate"))
    val d = if (decision == "none") chosen else decision
    all {
      decision' = d,
      if (d == "keep") all { pc' = pc.set(a, "done"), inflight' = inflight }
      else if (unapplied) all { pc' = pc.set(a, "done"), inflight' = inflight }
      else all {
        pc' = pc.set(a, "unapplying"), inflight' = inflight.union(Set((a, "unapply"))),
      },
    },
    answered' = answered, current' = current, applied' = applied, unapplied' = unapplied,
    unchangedErp,
  }

  action heardUnapply(a: int): bool = all {
    a == current, pc.get(a) == "unapplying", answered.contains((a, "unapply")),
    pc' = pc.set(a, "done"), unapplied' = true,
    inflight' = inflight, answered' = answered, current' = current, applied' = applied,
    decision' = decision, unchangedErp,
  }

  // The lease expires; the next attempt replays the checkpoints. The stale attempt's calls in
  // flight stay in flight, and it writes nothing more.
  action expire = all {
    current < 2, pc.get(current) != "done",
    current' = current + 1,
    pc' = pc, inflight' = inflight, answered' = answered, applied' = applied,
    decision' = decision, unapplied' = unapplied, unchangedErp,
  }

  // --- the ERP ---

  action land(c: Call): bool = all {
    inflight.contains(c),
    val a = c._1
    val op = c._2
    val fenced = not(buggyNoClaim) and (
      if (op == "apply") claims != Set() else claims.contains("undone"))
    all {
      matched' = if (fenced) matched else op == "apply",
      claims' = if (fenced or buggyNoClaim) claims
                 else claims.union(Set(if (op == "apply") "applied" else "undone")),
      compensated' = (compensated or (op == "unapply" and not(fenced) and matched)),
      staleLanded' = (staleLanded or a != current),
    },
    inflight' = inflight.exclude(Set(c)), answered' = answered.union(Set(c)),
    foreign' = foreign, current' = current, pc' = pc, applied' = applied, decision' = decision,
    unapplied' = unapplied,
  }

  action stutter = all {
    matched' = matched, foreign' = foreign, claims' = claims, inflight' = inflight,
    answered' = answered,
    current' = current, pc' = pc, applied' = applied, decision' = decision,
    unapplied' = unapplied, staleLanded' = staleLanded, compensated' = compensated,
  }

  action step = any {
    nondet a = oneOf(ATTEMPTS)
    any { start(a), heardApply(a), decide(a), heardUnapply(a) },
    expire,
    nondet c = oneOf(ATTEMPTS.map(a => (a, "apply")).union(ATTEMPTS.map(a => (a, "unapply"))))
    land(c),
    stutter,
  }

  // --- properties ---

  // Once the last attempt is done and nothing is in flight, the books are its decision.
  val settled: bool = pc.get(current) == "done" and inflight == Set()
  val booksAreTheDecision: bool = settled implies (matched == (decision == "keep"))

  // Witnesses: each must be violated, so the invariant is checked over runs that reach it.
  val neverStaleLanded: bool = not(staleLanded and settled)
  val neverCompensated: bool = not(compensated and settled)
  val neverForeignUndone: bool = not(foreign and compensated and settled)
}
