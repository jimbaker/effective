// Effective: race and quorum. The choice, the flag, and one global crash.
//
// Three branches race for `k` winners. An op is admitted, commits its ledger row, then commits
// its checkpoint, so the window between the row and the checkpoint exists. The parent saves the
// choice, then publishes a flag naming the losers; a flagged loser stops at its next admission,
// and an op already admitted runs to its checkpoint. A crash loses every ephemeral bit; recovery
// serves the checkpoints and republishes the flag from the saved choice before driving a branch.
//
// Each invariant is one ruling of race and quorum, read at the moment it can fail. Each toggle
// below is a rejected design, and flipping it must violate the invariant named beside it.

module race {
  pure val B: Set[int] = Set(0, 1, 2)
  pure val L: int = 2          // ops per branch
  pure val D: int = 2          // the deadline; the race starts at time 0
  pure val T: int = 3          // the clock's bound
  pure val MaxCrash: int = 1
  pure val INNER: int = 2      // the branch that holds a nested race

  // --- toggles: each false is the ruled design ---
  pure val flagBeforeChoice: bool = false            // everyStopHasItsChoice, noWinnerWasStopped
  pure val choiceInMemory: bool = false              // choiceStable
  pure val impossibleFromOutcomes: bool = false      // impossibleOnlyWhenHopeless
  pure val lateFlagCheck: bool = false               // noAdmitAfterFlag
  pure val recoveryDrivesBeforeChoice: bool = false  // noAdmitAfterFlag
  pure val abandonAdmitted: bool = false             // admittedRecordedAtBarrier
  pure val returnBeforeQuiescent: bool = false       // returnsQuiescent
  pure val innerIgnoresFlag: bool = false            // noInnerChoiceAfterOuterFlag
  pure val deadlineFromRestart: bool = false         // deadlineNotExtended
  pure val tieIsInTime: bool = false                 // winnersBeforeDeadline
  pure val earlyTimeout: bool = false                // timeoutOnlyAtDeadline
  pure val zeroStartsBranches: bool = false          // zeroStartsNothing

  // A choice's kind is "none" until saved, then "winners", "impossible" or "timeout".
  type Durable = {
    k: int,
    succeeds: int -> bool,     // each branch's ending, which replay reproduces
    chk: int -> int,           // checkpointed ops per branch
    led: Set[(int, int)],      // ledger rows, placed by (branch, op)
    doneAt: int -> int,        // the time a branch's last checkpoint landed; -1 while running
    kind: str,
    winners: Set[int],
    deadline: int,
    innerSaved: bool,
    returned: bool,
    meter: int,
    now: int,
  }

  type Ephemeral = {
    inflight: int -> int,      // 0 idle, 1 admitted, 2 ledger row committed
    published: bool,
    flag: Set[int],            // the losers the published flag names
    flagWinners: Set[int],     // the winners the flag was derived from
    stopped: Set[int],
    tentKind: str,             // flagBeforeChoice only: the choice before it is saved
    tentWinners: Set[int],
    tentS: int,
    tentU: int,
    admitted: Set[(int, int)], // ops admitted in this incarnation
  }

  type History = {
    crashes: int,
    published: bool,               // a flag has been published in some incarnation
    savedAtCrash: bool,            // a crash found a choice saved
    stops: Set[(int, Set[int])],   // each stop, with the winners of the flag that caused it
    lateAdmit: bool,               // a loser of the saved choice admitted under a published flag
    lateInner: bool,               // the inner race saved a choice under a published flag
    everAdmitted: Set[(int, int)],
    firstKind: str,
    firstWinners: Set[int],
    decS: int,                     // successes the parent saw when it decided
    decU: int,                     // branches unresolved when it decided
  }

  var d: Durable
  var e: Ephemeral
  var h: History

  pure val freshE: Ephemeral = {
    inflight: B.mapBy(_ => 0), published: false, flag: Set(), flagWinners: Set(),
    stopped: Set(), tentKind: "none", tentWinners: Set(), tentS: 0, tentU: 0, admitted: Set(),
  }

  // The n lowest members of s.
  pure def lowest(s: Set[int], n: int): Set[int] = s.filter(i => s.filter(j => j < i).size() < n)

  pure def losersOf(kind: str, w: Set[int]): Set[int] =
    if (kind == "winners") B.exclude(w) else B

  def running(i: int): bool = d.chk.get(i) < L

  val savedLosers: Set[int] = if (d.kind == "none") Set() else losersOf(d.kind, d.winners)

  // A flag is up in some incarnation, or recovery owes one: a crash found the choice saved.
  val flagOwed: bool = h.published or h.savedAtCrash

  val completed: Set[int] = B.filter(i => d.chk.get(i) == L)
  val eligible: Set[int] = completed.filter(i =>
    d.succeeds.get(i) and
    (if (tieIsInTime) d.doneAt.get(i) <= d.deadline else d.doneAt.get(i) < d.deadline))
  val unresolved: Set[int] = B.exclude(completed)
  val pastDeadline: bool =
    if (tieIsInTime) d.now > d.deadline
    else if (earlyTimeout) d.now >= d.deadline - 1
    else d.now >= d.deadline
  val hopeless: bool =
    if (impossibleFromOutcomes) B.filter(i => d.succeeds.get(i)).size() < d.k
    else eligible.size() + unresolved.size() < d.k

  // What the parent would save if it woke now.
  val nextKind: str =
    if (eligible.size() >= d.k) "winners"
    else if (pastDeadline) "timeout"
    else if (hopeless) "impossible"
    else "none"
  val nextWinners: Set[int] = if (nextKind == "winners") lowest(eligible, d.k) else Set()

  def saved(kind: str, w: Set[int], s: int, u: int): History =
    if (h.firstKind == "none") { ...h, firstKind: kind, firstWinners: w, decS: s, decU: u }
    else h

  action init = {
    nondet kk = oneOf(Set(0, 1, 2, 3))
    nondet ss = oneOf(B.setOfMaps(Set(true, false)))
    all {
      d' = {
        k: kk, succeeds: ss, chk: B.mapBy(_ => 0), led: Set(), doneAt: B.mapBy(_ => -1),
        kind: "none", winners: Set(), deadline: D, innerSaved: false, returned: false,
        meter: 0, now: 0,
      },
      e' = freshE,
      h' = {
        crashes: 0, published: false, savedAtCrash: false, stops: Set(), lateAdmit: false,
        lateInner: false,
        everAdmitted: Set(), firstKind: "none", firstWinners: Set(), decS: 0, decU: 0,
      },
    }
  }

  // --- a branch ---

  // `k = 0` is answered without starting a branch.
  action admit(i: int): bool = all {
    zeroStartsBranches or d.k > 0, not(d.returned), running(i), e.inflight.get(i) == 0,
    not(e.stopped.contains(i)),
    lateFlagCheck or not(e.flag.contains(i)),
    e' = { ...e, inflight: e.inflight.set(i, 1),
           admitted: e.admitted.union(Set((i, d.chk.get(i)))) },
    h' = { ...h, lateAdmit: h.lateAdmit or (flagOwed and savedLosers.contains(i)),
           everAdmitted: h.everAdmitted.union(Set((i, d.chk.get(i)))) },
    d' = d,
  }

  action commitRow(i: int): bool = all {
    e.inflight.get(i) == 1,
    d' = { ...d, led: d.led.union(Set((i, d.chk.get(i)))) },
    e' = { ...e, inflight: e.inflight.set(i, 2) },
    h' = h,
  }

  action commitCheckpoint(i: int): bool = all {
    e.inflight.get(i) == 2,
    val n = d.chk.get(i) + 1
    d' = { ...d, chk: d.chk.set(i, n),
           doneAt: if (n == L) d.doneAt.set(i, d.now) else d.doneAt },
    e' = { ...e, inflight: e.inflight.set(i, 0) },
    h' = h,
  }

  // A flagged loser stops at its next admission. Under abandonAdmitted it also drops an op
  // it had already admitted.
  action stop(i: int): bool = all {
    e.flag.contains(i), running(i), not(e.stopped.contains(i)),
    abandonAdmitted or e.inflight.get(i) == 0,
    e' = { ...e, stopped: e.stopped.union(Set(i)), inflight: e.inflight.set(i, 0) },
    h' = { ...h, stops: h.stops.union(Set((i, e.flagWinners))) },
    d' = d,
  }

  // --- the parent ---

  action decide = all {
    not(flagBeforeChoice), not(d.returned), d.kind == "none", nextKind != "none",
    d' = { ...d, kind: nextKind, winners: nextWinners },
    h' = saved(nextKind, nextWinners, eligible.size(), unresolved.size()),
    e' = e,
  }

  action publish = all {
    not(d.returned), d.kind != "none", not(e.published),
    e' = { ...e, published: true, flag: losersOf(d.kind, d.winners), flagWinners: d.winners },
    h' = { ...h, published: true },
    d' = d,
  }

  // flagBeforeChoice: decide in memory, publish the flag from that, save afterward.
  action decideTentative = all {
    flagBeforeChoice, not(d.returned), d.kind == "none", e.tentKind == "none",
    nextKind != "none",
    e' = { ...e, tentKind: nextKind, tentWinners: nextWinners, tentS: eligible.size(),
           tentU: unresolved.size() },
    d' = d, h' = h,
  }

  action publishTentative = all {
    flagBeforeChoice, e.tentKind != "none", not(e.published),
    e' = { ...e, published: true, flag: losersOf(e.tentKind, e.tentWinners),
           flagWinners: e.tentWinners },
    h' = { ...h, published: true },
    d' = d,
  }

  action saveTentative = all {
    flagBeforeChoice, d.kind == "none", e.tentKind != "none", e.published,
    d' = { ...d, kind: e.tentKind, winners: e.tentWinners },
    h' = saved(e.tentKind, e.tentWinners, e.tentS, e.tentU),
    e' = e,
  }

  // The nested race inside INNER saves its choice while INNER runs, unless INNER is flagged.
  action innerDecide = all {
    d.k > 0, not(d.innerSaved), running(INNER), not(e.stopped.contains(INNER)),
    innerIgnoresFlag or not(e.flag.contains(INNER)),
    d' = { ...d, innerSaved: true },
    h' = { ...h, lateInner: h.lateInner or (flagOwed and savedLosers.contains(INNER)) },
    e' = e,
  }

  action barrier = all {
    not(d.returned), d.kind != "none", e.published,
    returnBeforeQuiescent or
      B.forall(i => e.inflight.get(i) == 0 and (not(running(i)) or e.stopped.contains(i))),
    d' = { ...d, returned: true, meter: B.fold(0, (acc, i) => acc + d.chk.get(i)) },
    e' = e, h' = h,
  }

  // --- the world ---

  action tick = all {
    not(d.returned), d.now < T,
    d' = { ...d, now: d.now + 1 },
    e' = e, h' = h,
  }

  // Every ephemeral bit is lost. Recovery serves the checkpoints and, when a choice is saved,
  // publishes the flag from it before any branch is driven.
  action crash = all {
    not(d.returned), h.crashes < MaxCrash,
    val republish = d.kind != "none" and not(recoveryDrivesBeforeChoice)
    all {
      e' = if (republish)
             { ...freshE, published: true, flag: losersOf(d.kind, d.winners),
               flagWinners: d.winners }
           else freshE,
      d' = { ...d,
             kind: if (choiceInMemory) "none" else d.kind,
             winners: if (choiceInMemory) Set() else d.winners,
             deadline: if (deadlineFromRestart) d.now + D else d.deadline },
      h' = { ...h, crashes: h.crashes + 1, published: h.published or republish,
             savedAtCrash: h.savedAtCrash or d.kind != "none" },
    }
  }

  action stutter = all { d.returned, d' = d, e' = e, h' = h }

  action branchStep = {
    nondet i = oneOf(B)
    any { admit(i), commitRow(i), commitCheckpoint(i), stop(i) }
  }

  action progress = any {
    branchStep, decide, publish, decideTentative, publishTentative, saveTentative,
    innerDecide, barrier, tick,
  }

  action step = any { progress, crash, stutter }

  // --- invariants ---

  // once saved, the choice never changes.
  val choiceStable: bool =
    h.firstKind == "none" or (d.kind == h.firstKind and d.winners == h.firstWinners)

  // a loser stops only for the choice on record. The stop is what the world observed.
  val everyStopHasItsChoice: bool =
    h.stops.forall(s => d.kind != "none" and s._2 == d.winners)

  val noWinnerWasStopped: bool =
    d.kind != "winners" or h.stops.forall(s => not(d.winners.contains(s._1)))

  // exactly k winners, each a completed success; k = 0 starts nothing.
  val exactlyK: bool =
    d.kind != "winners" or
    (d.winners.size() == d.k and d.winners.forall(i => not(running(i)) and d.succeeds.get(i)))

  val zeroStartsNothing: bool = d.k != 0 or h.everAdmitted == Set()

  // impossible only when, at the decision, successes plus unresolved fell short of k.
  val impossibleOnlyWhenHopeless: bool = d.kind != "impossible" or h.decS + h.decU < d.k

  // no loser of the saved choice admits an op once a flag has been published, in any
  // incarnation, or after a crash that found the choice saved.
  val noAdmitAfterFlag: bool = not(h.lateAdmit)

  // every op admitted by the incarnation that returns has its checkpoint.
  val admittedRecordedAtBarrier: bool =
    not(d.returned) or e.admitted.forall(a => a._2 < d.chk.get(a._1))

  // `admittedRecordedAtBarrier` read over every incarnation, which a crash in flight after the
  // choice breaks.
  val everyAdmittedRecordedAtBarrier: bool =
    not(d.returned) or h.everAdmitted.forall(a => a._2 < d.chk.get(a._1))

  // the race returns only when no loser has an op in flight or can still admit one.
  val returnsQuiescent: bool =
    not(d.returned) or
    B.forall(i => e.inflight.get(i) == 0 and (not(running(i)) or e.stopped.contains(i)))

  // every checkpointed op, a loser's included, is folded at the barrier.
  val loserSpendReachesMeter: bool =
    not(d.returned) or d.meter == B.fold(0, (acc, i) => acc + d.chk.get(i))

  // The orphan row: a loser's row whose op a crash interrupted after the choice has no
  // checkpoint, which breaks this.
  val everyRowHasItsCheckpoint: bool =
    not(d.returned) or d.led.forall(r => r._2 < d.chk.get(r._1))

  // an inner race never saves a choice once its enclosing loser's flag was published.
  val noInnerChoiceAfterOuterFlag: bool = not(h.lateInner)

  // a restart never moves the deadline, a winner completed before it, and a timeout
  // comes at or after it.
  val deadlineNotExtended: bool = d.deadline <= D
  val winnersBeforeDeadline: bool =
    d.kind != "winners" or d.winners.forall(i => d.doneAt.get(i) < D)
  val timeoutOnlyAtDeadline: bool = d.kind != "timeout" or d.now >= D

  val safe: bool =
    choiceStable and everyStopHasItsChoice and noWinnerWasStopped and exactlyK and
    zeroStartsNothing and impossibleOnlyWhenHopeless and noAdmitAfterFlag and
    admittedRecordedAtBarrier and returnsQuiescent and loserSpendReachesMeter and
    noInnerChoiceAfterOuterFlag and deadlineNotExtended and winnersBeforeDeadline and
    timeoutOnlyAtDeadline

  // Reachability witnesses: each must be violated.
  val neverWinners: bool = d.kind != "winners"
  val neverImpossible: bool = d.kind != "impossible"
  val neverTimeout: bool = d.kind != "timeout"
  val neverZeroReturns: bool = not(d.returned and d.k == 0)
  val neverFullQuorumReturns: bool = not(d.returned and d.k == 3 and d.kind == "winners")

  // --- liveness ---

  val vars = (d, e, h)

  // A crash is possible, never obligatory, and bounded by MaxCrash.
  temporal fairness = weakFair(progress, vars)

  temporal liveness = fairness implies eventually(d.returned)

  // Expected VIOLATED: the counterexample is a fair run that returns.
  temporal nonVacuityWitness = fairness implies always(not(d.returned))
}
