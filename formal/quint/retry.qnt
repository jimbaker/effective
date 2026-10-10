// Effective: retries. A task fails at once on an error a retry would raise again.
//
// The workflow is one program chosen at init and held fixed across attempts, which is how the
// model states determinism: what user code raises is a function of the record. Each read of the
// tape is an effect whose outcome the environment chooses among the rows that read has in the
// handler's transition table: a hit, a miss whose thunk runs and writes, fails or loses its
// write, a store error, a park, a race's deadline read, and a wake race lost. A crash can end any
// cur. A run counts the thunks it ran and the substrate reads it counted; the edge fails a code
// error at once when both are zero, and otherwise retries while attempts remain.
//
// Each toggle below is a discarded rule, and flipping it must violate the invariant beside it.

module retry {
  pure val N: int = 2              // ops per program
  pure val MaxAttempts: int = 3
  pure val OPS: Set[int] = 0.to(N - 1)
  pure val KINDS: Set[str] = Set("step", "race", "wake")

  // --- toggles: each false is the ruled design ---
  pure val noFutileRule: bool = false            // noFutileAttemptBurned
  pure val staticWalkMark: bool = false          // noRecoverableTaskFailedEarly
  pure val compareSignatures: bool = false       // noRecoverableTaskFailedEarly
  pure val storeErrorsUncounted: bool = false    // noRecoverableTaskFailedEarly
  pure val raceClockUncounted: bool = false      // noRecoverableTaskFailedEarly
  pure val wakeRaceUncounted: bool = false       // noRecoverableTaskFailedEarly
  pure val thunkErrorCounted: bool = false       // countIsTheSubstrates
  pure val successfulThunkUncounted: bool = false // noRecoverableTaskFailedEarly

  type Run = {
    pc: int,
    ran: int,          // thunks this run ran
    read: int,         // substrate reads this run counted
    world: int,        // substrate reads this run met that answered other than the record
    thunked: bool,     // a thunk ran this run, whatever the counts say
  }

  var ops: List[str]
  var bug: bool             // user code raises after the last op when op 0's value is "bad"
  var tape: int -> str      // "" until written, then the value a retry is handed
  var handed: int -> str    // what user code was last handed per op, live or from the tape
  var cur: Run
  var attempt: int
  var status: str           // "running", "done", "failedEarly", "exhausted"
  var lastRaise: str        // what the last run to end raised: "", "crash", or a code error
  var burned: bool          // a retry followed user code's error from a run that ran no thunk
  var countedWorld: bool    // every run so far counted exactly the substrate reads it met

  pure val freshRun: Run = { pc: 0, ran: 0, read: 0, world: 0, thunked: false }

  action init = {
    nondet k0 = oneOf(KINDS)
    nondet k1 = oneOf(KINDS)
    nondet b = oneOf(Set(true, false))
    all {
      ops' = [k0, k1],
      bug' = b,
      tape' = OPS.mapBy(_ => ""),
      handed' = OPS.mapBy(_ => ""),
      cur' = freshRun,
      attempt' = 1,
      status' = "running",
      lastRaise' = "",
      burned' = false,
      countedWorld' = true,
    }
  }

  // The edge. A crash is no code error, so it retries while attempts remain.
  def failsEarly(raised: str, ended: Run): bool =
    raised != "crash" and (
      if (noFutileRule) false
      else if (compareSignatures) raised == lastRaise
      else if (staticWalkMark) ended.ran == 0
      else ended.ran == 0 and ended.read == 0)

  action endRun(raised: str, ended: Run): bool = {
    val early = failsEarly(raised, ended)
    val again = not(early) and attempt < MaxAttempts
    all {
      status' = if (early) "failedEarly" else if (again) "running" else "exhausted",
      attempt' = if (again) attempt + 1 else attempt,
      cur' = freshRun,
      lastRaise' = raised,
      burned' = (burned or (again and raised == "bug" and not(ended.thunked))),
      countedWorld' = (countedWorld and ended.read == ended.world),
      tape' = tape, handed' = handed, ops' = ops, bug' = bug,
    }
  }

  // A park ends the run, and the same attempt resumes as a fresh cur.
  action park: bool = all {
    cur' = freshRun,
    countedWorld' = (countedWorld and cur.read == cur.world),
    attempt' = attempt, status' = status, lastRaise' = lastRaise, burned' = burned,
    tape' = tape, handed' = handed, ops' = ops, bug' = bug,
  }

  // The op at `cur.pc` answers: the tape becomes `written`, and user code is handed `live`.
  action advance(written: int -> str, live: str, thunk: bool): bool = all {
    cur' = { ...cur, pc: cur.pc + 1,
             ran: cur.ran + (if (thunk and not(successfulThunkUncounted)) 1 else 0),
             thunked: (cur.thunked or thunk) },
    tape' = written,
    handed' = handed.set(cur.pc, live),
    attempt' = attempt, status' = status, lastRaise' = lastRaise, burned' = burned,
    countedWorld' = countedWorld, ops' = ops, bug' = bug,
  }

  // A substrate read that raises, counted or not.
  pure def met(r: Run, counted: bool): Run =
    { ...r, read: r.read + (if (counted) 1 else 0), world: r.world + 1 }

  // One read of op `cur.pc`'s record. Each `outcome` is a row of the transition table; a row
  // the op's kind lacks is disabled.
  action readTape = {
    val i = cur.pc
    val kind = ops[i]
    nondet v = oneOf(Set("good", "bad"))
    // A thunk's value reaches the tape as itself or as its canonical twin (a key order, a tuple
    // read back as a list), and user code is handed the value the thunk returned.
    nondet stored = oneOf(Set(v, "canon"))
    nondet outcome = oneOf(Set(
      "hit", "written", "thunkRaised", "writeLost", "storeRaised", "parked",
      "raceEarly", "raceLate", "wakeLost"))
    all {
      status == "running", i < N,
      (outcome == "hit") == (tape.get(i) != ""),
      if (outcome == "hit") advance(tape, tape.get(i), false)
      else if (outcome == "written" and kind == "wake") advance(tape.set(i, v), v, false)
      else if (outcome == "written") advance(tape.set(i, stored), v, true)
      else if (outcome == "thunkRaised")
        all { kind != "wake",
              endRun("fresh", { ...cur, ran: cur.ran + 1, thunked: true,
                                read: cur.read + (if (thunkErrorCounted) 1 else 0) }) }
      else if (outcome == "writeLost")
        all { kind == "step",
              endRun("store", met({ ...cur, ran: cur.ran + 1, thunked: true },
                                  not(storeErrorsUncounted))) }
      else if (outcome == "storeRaised") endRun("store", met(cur, not(storeErrorsUncounted)))
      else if (outcome == "parked") all { kind == "wake", park }
      else if (outcome == "raceEarly")
        all { kind == "race", endRun("raceFailed", met(cur, not(raceClockUncounted))) }
      else if (outcome == "raceLate")
        all { kind == "race", advance(tape.set(i, "timedOut"), "timedOut", false) }
      else all { kind == "wake", endRun("wakeLost", met(cur, not(wakeRaceUncounted))) }
    }
  }

  // Past the last op, user code returns, or raises on the value it was handed for op 0.
  action finish = all {
    status == "running", cur.pc == N,
    if (bug and handed.get(0) == "bad") endRun("bug", cur)
    else all {
      status' = "done",
      countedWorld' = (countedWorld and cur.read == cur.world),
      cur' = cur, attempt' = attempt, lastRaise' = lastRaise, burned' = burned,
      tape' = tape, handed' = handed, ops' = ops, bug' = bug,
    },
  }

  action crash = all { status == "running", endRun("crash", cur) }

  action stutter = all {
    status != "running",
    cur' = cur, attempt' = attempt, status' = status, lastRaise' = lastRaise, burned' = burned,
    countedWorld' = countedWorld, tape' = tape, handed' = handed, ops' = ops, bug' = bug,
  }

  action step = any { readTape, finish, crash, stutter }

  // --- invariants ---

  // The edge fails a task early only on an error every continuation raises again: user code's
  // own, on values that were each the tape's, which a retry is handed unchanged.
  val noRecoverableTaskFailedEarly: bool =
    status != "failedEarly" or
    (lastRaise == "bug" and OPS.forall(i => handed.get(i) == tape.get(i)))

  // No retry follows user code's error from a run that ran no thunk, whatever it counted.
  val noFutileAttemptBurned: bool = not(burned)

  // A run counts the substrate reads it met that answered other than the record, and nothing
  // else. Only a toggle parts the two, so this names what a tooth breaks.
  val countIsTheSubstrates: bool = countedWorld

  val safe: bool = noRecoverableTaskFailedEarly and noFutileAttemptBurned and countIsTheSubstrates

  // Reachability witnesses: each must be violated.
  val neverFailedEarly: bool = status != "failedEarly"
  val neverDone: bool = status != "done"
  val neverExhausted: bool = status != "exhausted"
  val neverRetriedAfterWorld: bool = not(attempt > 1 and lastRaise == "store")
}
