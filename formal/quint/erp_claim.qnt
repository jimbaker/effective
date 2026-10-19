// Effective: a settlement's apply and its undo at the ERP, across two attempts and two rounds.
//
// A round's settlement sends an Apply under the round's key and records the answer as its
// checkpoint. It then decides to keep the match or compensate. Keeping sends a reminder and ends
// the round completed; compensating sends an Unapply that names the round's Apply, and ends the
// round compensated. A retriable leg can run out, which ends the round unlanded with its call
// still in flight. Each ending is a ledger row. Round 1 opens once round 0 has ended compensated;
// a round that ended unlanded is continued by its next attempt, under its own key.
//
// An attempt's lease can expire at any point; the next attempt replays the round's checkpoints
// and runs the rest. A call already sent cannot be recalled, so it lands at the ERP whenever the
// network delivers it, whichever attempt sent it.
//
// The ERP keeps a claim on each round's key, written in the transaction of the match it records.
// An Apply finding its key claimed lands nothing; otherwise it claims the key as applied and
// records the match it made, or the match it found standing. An Unapply finding its key claimed
// as undone lands nothing; otherwise it claims the key so, and removes the match when its key's
// claim was the last live one recording it. The run may start with a match a person made.
//
// Assumed, as the engine's question: an attempt whose lease expired writes no checkpoint. The
// model writes a checkpoint only for the attempt that holds the lease. Assumed, as the ERP's: one
// landing is one whole fence run, which the fence makes true by serializing its runs per invoice.

module erp_claim {
  pure val ROUNDS: Set[int] = Set(0, 1)
  pure val ATTEMPTS: Set[int] = Set(1, 2)

  // --- the design: each field set false is a rule the design rejects ---
  pure val design = { claimed: true, counted: true, waits: true, continues: true }

  // A call in flight: its round, the attempt that sent it, and "apply" or "unapply".
  type Call = (int, int, str)

  // partial: the match standing, "none", "human", or the round whose Apply made it.
  // claim: each round's key, "none", "applied" or "undone".
  // pc: the current attempt's place, "start", "applying", "deciding", "unapplying", "reminding",
  // "stalled" or "done".
  // ending: each round's latest ended row, "none", "completed", "compensated" or "unlanded".
  var partial: str
  var claim: int -> str
  var holds: int -> str          // the match a round's applied claim recorded
  var inflight: Set[Call]
  var answered: Set[Call]
  var opened: int -> bool
  var current: int -> int        // the attempt holding the round's lease
  var pc: int -> str
  var applied: int -> bool       // checkpoint: the Apply's answer
  var decision: int -> str       // checkpoint: "none", "keep" or "compensate"
  var unapplied: int -> bool     // checkpoint: the Unapply's answer
  var ending: int -> str
  var staleLanded: bool          // a call from an attempt without the lease reached the ERP
  var compensated: bool          // an Unapply removed a match
  var foreignUndone: bool        // an Unapply removed the match a person made
  var continued: bool            // a round that ended unlanded was continued

  pure def made(r: int): str = if (r == 0) "r0" else "r1"

  action init = all {
    nondet standing = oneOf(Set("none", "human"))
    partial' = standing,
    claim' = ROUNDS.mapBy(_ => "none"), holds' = ROUNDS.mapBy(_ => "none"),
    inflight' = Set(), answered' = Set(),
    opened' = ROUNDS.mapBy(r => r == 0), current' = ROUNDS.mapBy(_ => 1),
    pc' = ROUNDS.mapBy(_ => "start"), applied' = ROUNDS.mapBy(_ => false),
    decision' = ROUNDS.mapBy(_ => "none"), unapplied' = ROUNDS.mapBy(_ => false),
    ending' = ROUNDS.mapBy(_ => "none"),
    staleLanded' = false, compensated' = false, foreignUndone' = false, continued' = false,
  }

  action unchangedErp = all {
    partial' = partial, claim' = claim, holds' = holds, staleLanded' = staleLanded,
    compensated' = compensated, foreignUndone' = foreignUndone,
  }

  action unchangedCheckpoints = all {
    applied' = applied, decision' = decision, unapplied' = unapplied,
  }

  // --- a round's current attempt ---

  action start(r: int): bool = all {
    opened.get(r), pc.get(r) == "start",
    if (applied.get(r)) all { pc' = pc.set(r, "deciding"), inflight' = inflight }
    else all {
      pc' = pc.set(r, "applying"),
      inflight' = inflight.union(Set((r, current.get(r), "apply"))),
    },
    answered' = answered, opened' = opened, current' = current, ending' = ending,
    continued' = continued, unchangedCheckpoints, unchangedErp,
  }

  action heardApply(r: int): bool = all {
    pc.get(r) == "applying", answered.contains((r, current.get(r), "apply")),
    pc' = pc.set(r, "deciding"), applied' = applied.set(r, true),
    inflight' = inflight, answered' = answered, opened' = opened, current' = current,
    decision' = decision, unapplied' = unapplied, ending' = ending, continued' = continued,
    unchangedErp,
  }

  action decide(r: int): bool = all {
    pc.get(r) == "deciding",
    nondet chosen = oneOf(Set("keep", "compensate"))
    val d = if (decision.get(r) == "none") chosen else decision.get(r)
    all {
      decision' = decision.set(r, d),
      if (d == "keep") all { pc' = pc.set(r, "reminding"), inflight' = inflight, ending' = ending }
      else if (unapplied.get(r)) all {
        pc' = pc.set(r, "done"), inflight' = inflight, ending' = ending.set(r, "compensated"),
      }
      else all {
        pc' = pc.set(r, "unapplying"), ending' = ending,
        inflight' = inflight.union(Set((r, current.get(r), "unapply"))),
      },
    },
    answered' = answered, opened' = opened, current' = current, applied' = applied,
    unapplied' = unapplied, continued' = continued, unchangedErp,
  }

  action heardUnapply(r: int): bool = all {
    pc.get(r) == "unapplying", answered.contains((r, current.get(r), "unapply")),
    pc' = pc.set(r, "done"), unapplied' = unapplied.set(r, true),
    ending' = ending.set(r, "compensated"),
    inflight' = inflight, answered' = answered, opened' = opened, current' = current,
    applied' = applied, decision' = decision, continued' = continued, unchangedErp,
  }

  action remind(r: int): bool = all {
    pc.get(r) == "reminding",
    pc' = pc.set(r, "done"), ending' = ending.set(r, "completed"),
    inflight' = inflight, answered' = answered, opened' = opened, current' = current,
    continued' = continued, unchangedCheckpoints, unchangedErp,
  }

  // A retriable leg runs out: the round ends unlanded, and its call stays in flight.
  action stall(r: int): bool = all {
    Set("applying", "unapplying", "reminding").contains(pc.get(r)),
    pc' = pc.set(r, "stalled"), ending' = ending.set(r, "unlanded"),
    inflight' = inflight, answered' = answered, opened' = opened, current' = current,
    continued' = continued, unchangedCheckpoints, unchangedErp,
  }

  // The lease expires, or a round that ended unlanded is continued: the next attempt replays the
  // round's checkpoints under the round's key. The earlier attempt's calls stay in flight, and it
  // writes nothing more.
  action expire(r: int): bool = all {
    current.get(r) < 2, pc.get(r) != "done",
    pc.get(r) != "stalled" or design.continues,
    current' = current.set(r, current.get(r) + 1), pc' = pc.set(r, "start"),
    continued' = (continued or pc.get(r) == "stalled"),
    inflight' = inflight, answered' = answered, opened' = opened, ending' = ending,
    unchangedCheckpoints, unchangedErp,
  }

  // --- the firm: round 1 opens ---

  val mayOpen: bool =
    if (not(design.waits)) true
    else if (design.continues) ending.get(0) == "compensated"
    else Set("compensated", "unlanded").contains(ending.get(0))

  action open = all {
    not(opened.get(1)), mayOpen,
    opened' = opened.set(1, true),
    pc' = pc, inflight' = inflight, answered' = answered, current' = current, ending' = ending,
    continued' = continued, unchangedCheckpoints, unchangedErp,
  }

  // --- the ERP ---

  action land(c: Call): bool = all {
    inflight.contains(c),
    val r = c._1
    val mine = claim.get(r)
    val fenced = design.claimed and (if (c._3 == "apply") mine != "none" else mine == "undone")
    val shared = ROUNDS.exists(o => o != r and claim.get(o) == "applied" and holds.get(o) == partial)
    val releases = not(design.counted) or (mine == "applied" and holds.get(r) == partial and not(shared))
    val removes = c._3 == "unapply" and not(fenced) and releases and partial != "none"
    val after = if (fenced) partial
                else if (c._3 == "apply") (if (partial == "none") made(r) else partial)
                else if (releases) "none" else partial
    all {
      partial' = after,
      claim' = if (fenced or not(design.claimed)) claim
               else claim.set(r, if (c._3 == "apply") "applied" else "undone"),
      holds' = if (fenced or c._3 != "apply") holds else holds.set(r, after),
      compensated' = (compensated or removes),
      foreignUndone' = (foreignUndone or (removes and partial == "human")),
      staleLanded' = (staleLanded or c._2 != current.get(r)),
    },
    inflight' = inflight.exclude(Set(c)), answered' = answered.union(Set(c)),
    opened' = opened, current' = current, pc' = pc, ending' = ending, continued' = continued,
    unchangedCheckpoints,
  }

  action stutter = all {
    inflight' = inflight, answered' = answered, opened' = opened, current' = current, pc' = pc,
    ending' = ending, continued' = continued, unchangedCheckpoints, unchangedErp,
  }

  action step = any {
    nondet r = oneOf(ROUNDS)
    any { start(r), heardApply(r), decide(r), heardUnapply(r), remind(r), stall(r), expire(r) },
    open,
    nondet c = oneOf(tuples(ROUNDS, ATTEMPTS, Set("apply", "unapply")))
    land(c),
    stutter,
  }

  // --- properties ---

  // Once every open round is done and nothing is in flight, the match stands exactly when a
  // round kept it.
  val settled: bool =
    inflight == Set() and ROUNDS.forall(r => opened.get(r) implies pc.get(r) == "done")
  val booksAreTheDecision: bool =
    settled implies ((partial != "none") == ROUNDS.exists(r => decision.get(r) == "keep"))

  // Every live applied claim's match stands.
  val liveClaimHasMatch: bool =
    ROUNDS.forall(r => claim.get(r) == "applied" implies holds.get(r) == partial)

  // A round opens only after the earlier round has ended.
  val noRoundBeforeEnded: bool = opened.get(1) implies ending.get(0) != "none"

  // At most one round keeps its credit.
  val creditedOnce: bool = ROUNDS.filter(r => decision.get(r) == "keep").size() <= 1

  // Witnesses: each must be violated, so the invariants are checked over runs that reach it.
  val neverStaleLanded: bool = not(staleLanded and settled)
  val neverCompensated: bool = not(compensated and settled)
  val neverForeignUndone: bool = not(foreignUndone and settled)
  val neverSecondRound: bool = not(opened.get(1) and settled)
  val neverContinued: bool = not(continued and settled)
}
