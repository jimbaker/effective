// Effective: an approval licenses what a settlement sends, across crashes, deploys and rounds.
//
// A worker resolving a round derives its terms under the code it runs, and the terms carry the
// digest of the settlement's plan. It parks on the round's approval, and the approver rules once
// per round. The gate hands the ruling on: an approval naming the plan the worker derived passes,
// and any other ruling stops the worker. A worker that passes claims the round; the first claim
// records its plan, and a later worker finds the round claimed and stops.
//
// The claim's holder sends the plan's legs in order: the memo and its apply, which can be undone;
// the notice, past which nothing compensates; and the reminder, which can run out of attempts and
// leave the round unlanded. A rejected leg before the notice turns the round to compensation, which
// undoes the landed legs latest first. A round that ended compensated opens the next one. A round
// that ended unlanded is continued once, by a later run that sends the reminder again.
//
// A crash drops a worker's place and keeps the round's checkpoints: the ruling, the claim, the
// legs landed, and the ending. The worker's process restarts on whatever code is deployed then,
// while the other worker keeps the code it started with, and a restarted worker derives its terms
// again and passes the gate again. A deploy can also find the case clean: its terms ask nobody, so
// the gate passes them unasked, and only the claim, recording the plan it was taken for, stands
// between the clean plan and an exception's approved round.
//
// Assumed: a leg sent and its answer are one step, so a crash lands between legs. A leg sent twice
// across a crash is the ERP's question, answered in erp_claim.qnt.

module plan_license {
  pure val ROUNDS: Set[int] = Set(0, 1)
  pure val WORKERS: Set[int] = Set(1, 2)
  pure val MAX_CRASHES: int = 2

  // --- the design: each field set false is a rule the design rejects ---
  pure val design = {
    bindRuling: true, checkContinuation: true, executorFromPlan: true, checkClaimedPlan: true,
  }

  // The code a worker runs: v1 derives and sends the approved plan, v2 a changed one, and v0 finds
  // the case clean and sends the clean plan, p0, which asks nobody. vx derives the approved plan
  // and sends another, an executor mapping its arguments outside the plan.
  pure val CODE: Set[str] =
    if (design.executorFromPlan) Set("v1", "v2", "v0") else Set("v1", "v2", "v0", "vx")
  pure def planOf(c: str): str = if (c == "v2") "p2" else if (c == "v0") "p0" else "p1"
  pure def sendsOf(c: str): str = if (c == "v1") "p1" else if (c == "v0") "p0" else "p2"
  pure def otherThan(p: str): str = if (p == "p1") "p2" else "p1"
  pure val CLEAN: str = "p0"

  // The legs in order: the memo, its apply, the notice, the reminder.
  pure val NOTICE: int = 2
  pure val REMINDER: int = 3
  pure val REMINDER_ATTEMPTS: int = 2

  // A leg sent: its round, the worker, the leg, "act", "undo" or "continue", and the plan sent.
  type Sent = (int, int, int, str, str)

  // verdict: "none", "approve", "deny", or "self" for an approval by the proposer.
  // names: the plan the ruling names, "" for none.
  type Ruling = { verdict: str, names: str }

  var code: int -> str           // the code a worker's process runs, chosen when it starts
  var crashes: int
  var opened: int -> bool
  var ruling: int -> Ruling
  var claimant: int -> int        // the worker whose claim won, 0 before any
  var claimed: int -> str         // the plan the winning claim recorded
  var authorized: int -> str      // the plan the ruling the claim rests on names
  var landed: int -> Set[int]     // checkpoint: the legs landed and not undone
  var compensating: int -> bool   // checkpoint: a leg was rejected
  var tries: int -> int           // checkpoint: the reminder's failed attempts
  // "none", "completed", "compensated", "unlanded", "failed" or "replanned"
  var ending: int -> str
  var continued: int -> bool
  var pc: int -> str              // a worker's place: "idle", "parked", "gated", "running", "done"
  var round: int -> int           // the round a worker resolves
  var derived: int -> str         // the plan the worker's code derived
  var sent: Set[Sent]
  var refusedOnReplay: bool       // a restarted worker's gate refused a ruling on another plan
  var superseded: bool            // a worker found its round claimed by the other
  var refusedAtClaim: bool        // a worker's own claim recorded another plan than it derived

  action init = all {
    code' = WORKERS.mapBy(_ => "v1"), crashes' = 0,
    opened' = ROUNDS.mapBy(r => r == 0),
    ruling' = ROUNDS.mapBy(_ => { verdict: "none", names: "" }),
    claimant' = ROUNDS.mapBy(_ => 0), claimed' = ROUNDS.mapBy(_ => ""),
    authorized' = ROUNDS.mapBy(_ => ""), landed' = ROUNDS.mapBy(_ => Set()),
    compensating' = ROUNDS.mapBy(_ => false), tries' = ROUNDS.mapBy(_ => 0),
    ending' = ROUNDS.mapBy(_ => "none"), continued' = ROUNDS.mapBy(_ => false),
    pc' = WORKERS.mapBy(_ => "idle"), round' = WORKERS.mapBy(_ => 0),
    derived' = WORKERS.mapBy(_ => ""), sent' = Set(),
    refusedOnReplay' = false, superseded' = false, refusedAtClaim' = false,
  }

  val active: int = if (opened.get(1)) 1 else 0

  action unchangedRounds = all {
    opened' = opened, ruling' = ruling, claimant' = claimant, claimed' = claimed,
    authorized' = authorized, landed' = landed, compensating' = compensating, tries' = tries,
    ending' = ending, continued' = continued,
  }

  action unchangedFlags = all {
    refusedOnReplay' = refusedOnReplay, superseded' = superseded, refusedAtClaim' = refusedAtClaim,
  }

  // --- a worker ---

  // Derive the terms under the running code and park on the round's approval.
  action begin(w: int): bool = all {
    pc.get(w) == "idle" or (pc.get(w) == "done" and round.get(w) < active),
    ending.get(active) == "none",
    pc' = pc.set(w, "parked"), round' = round.set(w, active),
    derived' = derived.set(w, planOf(code.get(w))),
    code' = code, crashes' = crashes, sent' = sent, unchangedRounds, unchangedFlags,
  }

  // The gate reads the round's ruling against the plan this worker derived. Clean terms ask
  // nobody, so they pass whatever the ruling says.
  action gate(w: int): bool = all {
    pc.get(w) == "parked",
    val r = round.get(w)
    val v = ruling.get(r)
    val clean = derived.get(w) == CLEAN
    val bound = not(design.bindRuling) or v.names == derived.get(w)
    all {
      clean or v.verdict != "none",
      pc' = pc.set(w, if (clean or (v.verdict == "approve" and bound)) "gated" else "done"),
      refusedOnReplay' = (refusedOnReplay
        or (not(clean) and crashes > 0 and v.verdict == "approve" and not(bound)
            and v.names != "")),
    },
    round' = round, derived' = derived, code' = code, crashes' = crashes, sent' = sent,
    superseded' = superseded, refusedAtClaim' = refusedAtClaim, unchangedRounds,
  }

  // The first claim records the plan it was taken for, and the ruling it rests on; a clean claim
  // rests on none. A worker finding its own claim compares the plan it derived with that one.
  action claim(w: int): bool = all {
    pc.get(w) == "gated",
    val r = round.get(w)
    val holder = claimant.get(r)
    val mine = holder == w
    val replanned = mine and design.checkClaimedPlan and claimed.get(r) != derived.get(w)
    val clean = derived.get(w) == CLEAN
    all {
      claimant' = if (holder == 0) claimant.set(r, w) else claimant,
      claimed' = if (holder == 0) claimed.set(r, derived.get(w)) else claimed,
      authorized' = if (holder == 0) authorized.set(r, if (clean) "" else ruling.get(r).names)
                    else authorized,
      pc' = pc.set(w, if (holder == 0 or (mine and not(replanned))) "running" else "done"),
      superseded' = (superseded or (holder != 0 and not(mine))),
      refusedAtClaim' = (refusedAtClaim or replanned),
    },
    round' = round, derived' = derived, code' = code, crashes' = crashes, sent' = sent,
    opened' = opened, ruling' = ruling, landed' = landed, compensating' = compensating,
    tries' = tries, ending' = ending, continued' = continued, refusedOnReplay' = refusedOnReplay,
  }

  pure def nextLeg(done: Set[int]): int =
    if (not(done.contains(0))) 0
    else if (not(done.contains(1))) 1
    else if (not(done.contains(NOTICE))) NOTICE
    else REMINDER

  // Send the next leg of the plan, as the running code's executor selects it.
  action act(w: int): bool = all {
    pc.get(w) == "running",
    val r = round.get(w)
    val leg = nextLeg(landed.get(r))
    all {
      ending.get(r) == "none", not(compensating.get(r)),
      sent' = sent.union(Set((r, w, leg, "act", sendsOf(code.get(w))))),
      nondet lands = oneOf(Set(true, false))
      val failedTries = tries.get(r) + 1
      val runsOut = leg == REMINDER and not(lands) and failedTries >= REMINDER_ATTEMPTS
      val finishes = leg == REMINDER and lands
      all {
        landed' = if (lands) landed.set(r, landed.get(r).union(Set(leg))) else landed,
        compensating' = if (not(lands) and leg < REMINDER) compensating.set(r, true)
                        else compensating,
        tries' = if (not(lands) and leg == REMINDER) tries.set(r, failedTries) else tries,
        ending' = if (finishes) ending.set(r, "completed")
                  else if (runsOut) ending.set(r, "unlanded")
                  else ending,
        pc' = if (finishes or runsOut) pc.set(w, "done") else pc,
      },
    },
    round' = round, derived' = derived, code' = code, crashes' = crashes,
    opened' = opened, ruling' = ruling, claimant' = claimant, claimed' = claimed,
    authorized' = authorized, continued' = continued, unchangedFlags,
  }

  // Undo the latest landed leg the notice has not passed; with none left, the round ends.
  action undo(w: int): bool = all {
    pc.get(w) == "running",
    val r = round.get(w)
    val done = landed.get(r)
    val latest = if (done.contains(1)) 1 else if (done.contains(0)) 0 else -1
    all {
      ending.get(r) == "none", compensating.get(r),
      if (latest == -1) all {
        ending' = ending.set(r, "compensated"), pc' = pc.set(w, "done"), sent' = sent,
        landed' = landed,
      } else all {
        sent' = sent.union(Set((r, w, latest, "undo", sendsOf(code.get(w))))),
        nondet undone = oneOf(Set(true, false))
        all {
          landed' = if (undone) landed.set(r, done.exclude(Set(latest))) else landed,
          ending' = if (undone) ending else ending.set(r, "failed"),
          pc' = if (undone) pc else pc.set(w, "done"),
        },
      },
    },
    round' = round, derived' = derived, code' = code, crashes' = crashes,
    opened' = opened, ruling' = ruling, claimant' = claimant, claimed' = claimed,
    authorized' = authorized, compensating' = compensating, tries' = tries,
    continued' = continued, unchangedFlags,
  }

  // A crash drops the worker's place, and its process restarts on whatever code is deployed.
  action crash(w: int): bool = all {
    crashes < MAX_CRASHES, Set("parked", "gated", "running").contains(pc.get(w)),
    nondet deployed = oneOf(CODE)
    code' = code.set(w, deployed),
    crashes' = crashes + 1, pc' = pc.set(w, "idle"),
    round' = round, derived' = derived, sent' = sent, unchangedRounds, unchangedFlags,
  }

  // A later run continues a round that ended unlanded, sending its reminder once more.
  action carryOn(w: int, r: int): bool = all {
    Set("idle", "done").contains(pc.get(w)),
    ending.get(r) == "unlanded", not(continued.get(r)),
    continued' = continued.set(r, true),
    if (design.checkContinuation and planOf(code.get(w)) != claimed.get(r)) all {
      ending' = ending.set(r, "replanned"), sent' = sent, landed' = landed,
    } else all {
      sent' = sent.union(Set((r, w, REMINDER, "continue", sendsOf(code.get(w))))),
      nondet lands = oneOf(Set(true, false))
      all {
        landed' = if (lands) landed.set(r, landed.get(r).union(Set(REMINDER))) else landed,
        ending' = if (lands) ending.set(r, "completed") else ending,
      },
    },
    pc' = pc, round' = round, derived' = derived, code' = code, crashes' = crashes,
    opened' = opened, ruling' = ruling, claimant' = claimant, claimed' = claimed,
    authorized' = authorized, compensating' = compensating, tries' = tries, unchangedFlags,
  }

  // --- the approver, and the firm ---

  // The approver rules once per round, shown the terms a parked worker derived.
  action rule(w: int): bool = all {
    pc.get(w) == "parked",
    val r = round.get(w)
    val shown = derived.get(w)
    all {
      ruling.get(r).verdict == "none", shown != CLEAN,
      nondet chosen = oneOf(Set(
        { verdict: "approve", names: shown },
        { verdict: "approve", names: otherThan(shown) },
        { verdict: "approve", names: "" },
        { verdict: "deny", names: shown },
        { verdict: "self", names: shown },
      ))
      ruling' = ruling.set(r, chosen),
    },
    opened' = opened, claimant' = claimant, claimed' = claimed, authorized' = authorized,
    landed' = landed, compensating' = compensating, tries' = tries, ending' = ending,
    continued' = continued, pc' = pc, round' = round, derived' = derived, code' = code,
    crashes' = crashes, sent' = sent, unchangedFlags,
  }

  // A round that ended compensated opens the next.
  action open = all {
    not(opened.get(1)), ending.get(0) == "compensated",
    opened' = opened.set(1, true),
    ruling' = ruling, claimant' = claimant, claimed' = claimed, authorized' = authorized,
    landed' = landed, compensating' = compensating, tries' = tries, ending' = ending,
    continued' = continued, pc' = pc, round' = round, derived' = derived, code' = code,
    crashes' = crashes, sent' = sent, unchangedFlags,
  }

  action stutter = all {
    code' = code, crashes' = crashes, pc' = pc, round' = round, derived' = derived,
    sent' = sent, unchangedRounds, unchangedFlags,
  }

  action step = any {
    nondet w = oneOf(WORKERS)
    any {
      begin(w), gate(w), claim(w), act(w), undo(w), crash(w), rule(w),
      nondet r = oneOf(ROUNDS)
      carryOn(w, r),
    },
    open,
    stutter,
  }

  // --- properties ---

  // Every leg sent sends the plan the round's claim recorded, and, for an exception, the plan
  // the ruling the claim rests on named; a clean plan is licensed by the policy's automatic
  // permission. An act or an undo is sent by the claim's holder.
  val licensed: bool = sent.forall(s =>
    s._5 == claimed.get(s._1) and (s._5 == CLEAN or s._5 == authorized.get(s._1))
      and (s._4 == "continue" or s._2 == claimant.get(s._1)))

  // An exception's round is claimed only on an approval, never on a denial or the proposer's own.
  val approvedByAnother: bool = ROUNDS.forall(r =>
    (claimant.get(r) != 0 and claimed.get(r) != CLEAN) implies ruling.get(r).verdict == "approve")

  // Witnesses: each must be violated, so the invariants are checked over runs that reach it.
  val neverSent: bool = sent == Set()
  val neverCompensated: bool = ROUNDS.forall(r => ending.get(r) != "compensated")
  val neverRefusedOnReplay: bool = not(refusedOnReplay)
  val neverReplanned: bool = ROUNDS.forall(r => ending.get(r) != "replanned")
  val neverContinuedLanded: bool =
    ROUNDS.forall(r => not(continued.get(r) and ending.get(r) == "completed"))
  val neverSecondRoundSent: bool = not(sent.exists(s => s._1 == 1))
  val neverSuperseded: bool = not(superseded)
  val neverRefusedAtClaim: bool = not(refusedAtClaim)
  val neverCleanSent: bool = not(sent.exists(s => s._5 == CLEAN))
}
