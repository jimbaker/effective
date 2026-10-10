// Effective: a race's branch errors, before and after the choice, across task retries.
//
// Three branches race for `k` winners. Each branch is one of six programs:
//
// | program  | on each attempt                                                                  |
// |----------|----------------------------------------------------------------------------------|
// | `ok`     | admits its op and succeeds                                                       |
// | `flaky`  | raises a retryable error before its op on the first attempt it runs, then is ok  |
// | `code`   | raises a code error before its op, having run nothing fresh, so a retry repeats it |
// | `typed`  | raises an `Unretryable` error before its op                                      |
// | `refuse` | admits its op, which the domain refuses                                          |
// | `late`   | admits its op, which runs and is recorded, then raises a code error; a retry      |
// |          | serves the op and raises before reaching another admission                         |
//
// A reader takes the branches' endings into the choice in batches: the in-order reader runs
// one branch at a time and reads each as it ends, and the concurrent reader runs them all and
// reads whatever has ended at each wake. A branch that reaches its op's admission after the
// choice is saved, as a loser, stops there, so a race of winners can end a branch before it
// raises. A choice of no winners waits for every branch to end, so no branch is stopped before it
// could raise and the race fails when one did. At the barrier an attempt fails when the race
// failed, when a loser raised a retryable error after a choice of winners, or when any branch
// raised after a choice of no winners. A failed attempt retries while attempts remain, keeping
// the saved choice and the world (a flaky branch's failure has happened, a late branch's op is on
// record). The race's deadline, its flag and crashes are `race.qnt`'s.
//
// The model runs the race twice: first on the programs in a canonical order with the in-order
// reader, then on a permutation of them with either reader, so an invariant can compare the two.

module race_errors {
  pure val B: Set[int] = Set(0, 1, 2)
  pure val N: int = 3
  pure val Limit: int = 3
  pure val KINDS: Set[str] = Set("ok", "flaky", "code", "typed", "refuse", "late")
  pure val RANK: str -> int =
    Map("ok" -> 0, "flaky" -> 1, "code" -> 2, "typed" -> 3, "refuse" -> 4, "late" -> 5)
  pure val LABELS: Set[str] = Set("won", "unchosen", "refusal", "raised", "stopped")

  // --- toggles: each false is the ruled design ---
  pure val earlierRule: bool = false        // orderIndependent, aSucceedingBranchWins
  pure val clearableIsEnding: bool = false  // noClearableErrorIsAnEnding
  pure val impossibleAtOnce: bool = false   // orderIndependent

  // A slot: a branch yet to start, at its op's admission, with its op admitted, or ended.
  pure val LIVE: Set[str] = Set("pending", "admitting", "admitted")
  pure val RAISED: Set[str] = Set("raisedRetryable", "raisedReplays", "raisedTyped")

  type Lane = {
    progs: int -> str,
    inOrder: bool,
    k: int,
    attempt: int,
    kind: str,              // the saved choice: "none", "winners" or "impossible"
    winners: Set[int],
    choseOn: int,           // the attempt that saved the choice; 0 while none is saved
    choseRunning: int,      // the branches still running when the choice was saved
    fired: Set[int],        // flaky branches whose one failure has happened
    recorded: Set[int],     // late branches whose op is on record
    slot: int -> str,
    read: Set[int],         // branches this attempt's reader has taken in
    before: Set[int],       // raised branches read before the choice
    failing: bool,          // the race failed: no branch was left to win
    status: str,            // "running", "answered" or "failed"
  }

  type Outcome = {
    status: str,
    kind: str,
    choseOn: int,
    attempt: int,
    endings: (str, str) -> int,
  }

  var lane: Lane
  var first: Outcome
  var phase: int            // 0 while the canonical run races, 1 while the permuted run does
  var perm: int -> int
  var secondInOrder: bool

  pure def ended(s: str): bool = not(LIVE.contains(s))
  pure def raised(s: str): bool = RAISED.contains(s)

  // The n lowest members of s.
  pure def lowest(s: Set[int], n: int): Set[int] = s.filter(i => s.filter(j => j < i).size() < n)

  pure def laneOf(progs: int -> str, inOrder: bool, k: int): Lane = {
    progs: progs, inOrder: inOrder, k: k, attempt: 1, kind: "none", winners: Set(),
    choseOn: 0, choseRunning: 0, fired: Set(), recorded: Set(), slot: B.mapBy(_ => "pending"), read: Set(), before: Set(), failing: false,
    status: "running",
  }

  // A branch's next phase. A loser reaching its admission after the choice stops there.
  pure def advance(l: Lane, i: int): Lane = {
    val s = l.slot.get(i)
    val p = l.progs.get(i)
    if (s == "pending")
      if (p == "code") { ...l, slot: l.slot.set(i, "raisedReplays") }
      else if (p == "typed") { ...l, slot: l.slot.set(i, "raisedTyped") }
      else if (p == "late" and l.recorded.contains(i)) { ...l, slot: l.slot.set(i, "raisedReplays") }
      else if (p == "flaky" and not(l.fired.contains(i)))
        { ...l, slot: l.slot.set(i, "raisedRetryable"), fired: l.fired.union(Set(i)) }
      else { ...l, slot: l.slot.set(i, "admitting") }
    else if (s == "admitting")
      if (l.kind != "none" and not(l.winners.contains(i))) { ...l, slot: l.slot.set(i, "stopped") }
      else if (p == "refuse") { ...l, slot: l.slot.set(i, "refused") }
      else if (p == "late") { ...l, slot: l.slot.set(i, "admitted"), recorded: l.recorded.union(Set(i)) }
      else { ...l, slot: l.slot.set(i, "admitted") }
    else if (p == "late") { ...l, slot: l.slot.set(i, "raisedReplays") }
    else { ...l, slot: l.slot.set(i, "won") }
  }

  // The in-order reader starts a branch once every branch before it is read.
  pure def canStep(l: Lane, i: int): bool =
    l.status == "running" and LIVE.contains(l.slot.get(i)) and
    (not(l.inOrder) or B.forall(j => j < i implies l.read.contains(j)))

  // One batch, everything ended and unread, taken into the choice. Before the choice a raised
  // branch is a loss; the race fails when it can choose no winners and a branch raised. A choice
  // of no winners waits until no branch runs.
  pure def wake(l: Lane): Lane = {
    val batch = B.filter(i => ended(l.slot.get(i)) and not(l.read.contains(i)))
    val read2 = l.read.union(batch)
    if (l.failing or l.kind != "none") { ...l, read: read2 }
    else {
      val before2 = l.before.union(batch.filter(i => raised(l.slot.get(i))))
      val earlier = l.read.filter(i => l.slot.get(i) == "won")
      val won = batch.filter(i => l.slot.get(i) == "won")
      val running = N - read2.size()
      val reached = earlier.size() + won.size() >= l.k
      val hopeless =
        if (impossibleAtOnce) earlier.size() + won.size() + running < l.k
        else running == 0 and earlier.size() + won.size() < l.k
      val fails =
        if (earlierRule) before2 != Set() else not(reached) and hopeless and before2 != Set()
      if (fails) { ...l, read: read2, before: before2, failing: true }
      else if (reached)
        { ...l, read: read2, before: before2, kind: "winners", choseOn: l.attempt,
          winners: earlier.union(lowest(won, l.k - earlier.size())) }
      else if (hopeless)
        { ...l, read: read2, before: before2, kind: "impossible", choseOn: l.attempt,
          choseRunning: running }
      else { ...l, read: read2, before: before2 }
    }
  }

  // A loser's retryable error after the choice: a retry can clear it.
  pure def clearable(l: Lane): bool =
    B.exists(i => l.slot.get(i) == "raisedRetryable" and not(l.before.contains(i)))

  // A branch raised after a choice of no winners, which it could have failed had it been read
  // first.
  pure def raisedAfterNoWinners(l: Lane): bool =
    l.kind == "impossible" and B.exists(i => raised(l.slot.get(i)) and not(l.before.contains(i)))

  pure def atBarrier(l: Lane): bool =
    l.status == "running" and B.forall(i => ended(l.slot.get(i)) and l.read.contains(i))

  pure def barrier(l: Lane): Lane =
    if (l.failing or (not(clearableIsEnding) and clearable(l)) or raisedAfterNoWinners(l))
      if (l.attempt < Limit)
        { ...l, attempt: l.attempt + 1, slot: B.mapBy(_ => "pending"), read: Set(),
          before: Set(), failing: false }
      else { ...l, status: "failed" }
    else { ...l, status: "answered" }

  pure def label(l: Lane, i: int): str = {
    val s = l.slot.get(i)
    if (raised(s)) "raised"
    else if (s == "stopped") "stopped"
    else if (s == "refused") "refusal"
    else if (l.winners.contains(i)) "won"
    else "unchosen"
  }

  pure def outcome(l: Lane): Outcome = {
    status: l.status,
    kind: l.kind,
    choseOn: l.choseOn,
    attempt: l.attempt,
    endings: tuples(KINDS, LABELS).mapBy(pl =>
      B.filter(i => l.progs.get(i) == pl._1 and label(l, i) == pl._2).size()),
  }

  pure def okCount(l: Lane): int = B.filter(i => l.progs.get(i) == "ok").size()

  // Branches that succeed on some attempt.
  pure def canSucceed(l: Lane): int =
    B.filter(i => l.progs.get(i) == "ok" or l.progs.get(i) == "flaky").size()

  pure val PERMS: Set[int -> int] = B.setOfMaps(B).filter(m => B.map(i => m.get(i)) == B)

  pure val noOutcome: Outcome = {
    status: "running", kind: "none", choseOn: 0, attempt: 0,
    endings: tuples(KINDS, LABELS).mapBy(_ => 0),
  }

  action init = {
    nondet progs = oneOf(B.setOfMaps(KINDS).filter(m =>
      RANK.get(m.get(0)) <= RANK.get(m.get(1)) and RANK.get(m.get(1)) <= RANK.get(m.get(2))))
    nondet kk = oneOf(Set(1, 2, 3))
    nondet pp = oneOf(PERMS)
    nondet io = oneOf(Set(true, false))
    all {
      lane' = laneOf(progs, true, kk),
      first' = noOutcome,
      phase' = 0,
      perm' = pp,
      secondInOrder' = io,
    }
  }

  action branchStep = {
    nondet i = oneOf(B)
    all {
      canStep(lane, i),
      lane' = advance(lane, i),
      first' = first, phase' = phase, perm' = perm, secondInOrder' = secondInOrder,
    }
  }

  action read = all {
    lane.status == "running",
    B.exists(i => ended(lane.slot.get(i)) and not(lane.read.contains(i))),
    lane' = wake(lane),
    first' = first, phase' = phase, perm' = perm, secondInOrder' = secondInOrder,
  }

  action finishAttempt = all {
    atBarrier(lane),
    lane' = barrier(lane),
    first' = first, phase' = phase, perm' = perm, secondInOrder' = secondInOrder,
  }

  // The canonical run has ended: race the permuted programs with the second reader.
  action raceAgain = all {
    phase == 0, lane.status != "running",
    first' = outcome(lane),
    lane' = laneOf(B.mapBy(j => lane.progs.get(perm.get(j))), secondInOrder, lane.k),
    phase' = 1,
    perm' = perm, secondInOrder' = secondInOrder,
  }

  action stutter = all {
    phase == 1, lane.status != "running",
    lane' = lane, first' = first, phase' = phase, perm' = perm, secondInOrder' = secondInOrder,
  }

  action step = any { branchStep, read, finishAttempt, raceAgain, stutter }

  // --- invariants ---

  val bothEnded: bool = phase == 1 and lane.status != "running"

  // The order the branches are written in, and the reader, change neither whether the task
  // answers, nor the kind of choice it answers with, nor the attempt that saves a choice of
  // winners, so the same branches can win. Which of several branches succeeding on that attempt
  // wins is the reader's: the lowest index in order, the first batch concurrently.
  val orderIndependent: bool =
    bothEnded implies (
      first.status == outcome(lane).status and
      (first.status == "answered" implies (
        first.kind == outcome(lane).kind and
        (first.kind == "winners" implies first.choseOn == outcome(lane).choseOn))))

  // The attempt that answers moves by at most one between orders and readers.
  val answeringAttemptWithinOne: bool =
    (bothEnded and first.status == "answered" and lane.status == "answered") implies (
      first.attempt <= outcome(lane).attempt + 1 and outcome(lane).attempt <= first.attempt + 1)

  // With at least `k` branches that succeed whenever they run, the task answers with winners.
  val aSucceedingBranchWins: bool =
    okCount(lane) >= lane.k implies (
      lane.status != "failed" and (lane.status == "answered" implies lane.kind == "winners"))

  // A race answers impossible only when fewer than `k` branches succeed on any attempt, so no
  // retry could have answered with winners.
  val impossibleOnlyWhenHopeless: bool =
    (lane.status == "answered" and lane.kind == "impossible") implies canSucceed(lane) < lane.k

  // A choice of no winners is saved only once every branch has ended, so none was stopped
  // short of a raise.
  val impossibleOnlyOnceEveryBranchEnded: bool = lane.kind == "impossible" implies lane.choseRunning == 0

  // No race answers holding a loser's error that a retry could clear.
  val noClearableErrorIsAnEnding: bool = lane.status == "answered" implies not(clearable(lane))

  // A choice of winners holds exactly `k`, each one a branch that succeeded.
  val winnersSucceeded: bool =
    (lane.status == "answered" and lane.kind == "winners") implies (
      lane.winners.size() == lane.k and lane.winners.forall(i => lane.slot.get(i) == "won"))

  val safe: bool =
    orderIndependent and answeringAttemptWithinOne and aSucceedingBranchWins and
    impossibleOnlyWhenHopeless and impossibleOnlyOnceEveryBranchEnded and
    noClearableErrorIsAnEnding and winnersSucceeded

  // Strongest readings, each expected to be violated: the order can move the attempt that
  // answers, and a loser read before the choice ends `raised` where one read after it can
  // end `stopped`.
  val sameAnsweringAttempt: bool = bothEnded implies first.attempt == outcome(lane).attempt
  val sameEndings: bool = bothEnded implies first.endings == outcome(lane).endings

  // Reachability witnesses: each must be violated.
  val neverBothEnded: bool = not(bothEnded)
  val neverImpossible: bool = not(lane.status == "answered" and lane.kind == "impossible")
  val neverTaskFailed: bool = lane.status != "failed"
  val neverAnsweredOnRetry: bool = not(lane.status == "answered" and lane.attempt > 1)
  val neverRaisedEnding: bool =
    not(lane.status == "answered" and B.exists(i => label(lane, i) == "raised"))
  val neverStoppedEnding: bool =
    not(lane.status == "answered" and B.exists(i => label(lane, i) == "stopped"))
}
