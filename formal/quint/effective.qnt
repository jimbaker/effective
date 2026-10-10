// Effective — the rules track in Quint (the reachability/interleaving half)
//
// Companion to the Lean development (`formal/lean/`).
//
// Division of labor: Lean proves the *pure* facts — name injectivity (T1),
// the transition relation is a partial function (T2/determinism), the ledger is
// append-only (T3). Quint explores what Lean punted on: the workflow continuation W
// and the reachable state space across interleavings. It *imports* the Lean facts as
// modeling choices (a deterministic step; abstract checkpoint names) rather than
// re-deriving them at every state.
//
// This is v0 — the smallest useful model: one straight-line workflow, the three
// guard-bearing ops (step / await / ledger), and one invariant with teeth
// (`ledgerNodup`). It exists to stand up the harness and prove the rung-1 discipline
// works in Quint (break the model, watch the checker flag it). Gather + the
// totality/suspend-in-gather check come next.

module effective {
  // A workflow op. Names/events/eids are abstract ints (Lean owns their injectivity).
  type Op =
    | Step(int)     // checkpoint step `n`; its effect is recorded once, replayed after
    | Await(int)    // suspend until event `e` is delivered
    | Ledger(int)   // append event-id `eid` to the ledger (idempotent by eid)

  // --- state: the part of the configuration the rules read/write ---
  var prog: List[Op]   // the workflow continuation W — remaining ops, in order
  var chk: Set[int]    // recorded checkpoint names (values abstracted away)
  var led: List[int]   // the ledger: committed event-ids, append-only
  var evt: Set[int]    // delivered events

  // The workflow under test: commit 10, await 5, commit 10 again (idempotent),
  // then a repeated step (replay). Reaching the second Ledger(10) requires event 5
  // to be delivered first — a genuine multi-step reachability check.
  pure val workflow: List[Op] = [Step(1), Ledger(10), Await(5), Ledger(10), Step(1)]

  // The universe of events the environment may deliver.
  pure val events: Set[int] = Set(5, 6)

  // BUG TOGGLE (the rung-1 experiment): when true, the ledger-append rule ignores its
  // `eid not already present` guard — i.e. drops Ledger-Idem. Flip to true and
  // `quint verify` should find a run where the ledger gets a duplicate (nodup fails).
  pure val buggyLedger: bool = false

  pure def ledHas(l: List[int], x: int): bool =
    l.indices().exists(i => l[i] == x)

  action init = all {
    prog' = workflow,
    chk'  = Set(),
    led'  = [],
    evt'  = Set(),
  }

  // Advance the head op (enabled only when it actually has an applicable rule).
  action stepHead = {
    val op = prog.head()
    val rest = prog.tail()
    match op {
      // Step-Run records the name; Step-Replay (name already in chk) leaves chk
      // unchanged. `union` is idempotent, so one clause models both — and that the
      // outcome is the same is exactly Lean's determinism, imported, not re-checked.
      | Step(n) => all {
          chk'  = chk.union(Set(n)),
          prog' = rest,
          led'  = led,
          evt'  = evt,
        }
      // Ledger-Append (eid fresh) vs Ledger-Idem (eid present). The guard is the
      // whole point; `buggyLedger` removes it to give the invariant something to catch.
      | Ledger(eid) => all {
          led'  = if (buggyLedger or not(ledHas(led, eid))) led.append(eid) else led,
          prog' = rest,
          chk'  = chk,
          evt'  = evt,
        }
      // Await-Resume: only enabled once the event is delivered. (Await-Park is the
      // *absence* of this transition — the environment must deliver `e` first.)
      | Await(e) => all {
          evt.contains(e),
          prog' = rest,
          chk'  = chk,
          led'  = led,
          evt'  = evt,
        }
    }
  }

  // The environment delivers an event (always available — models the outside world).
  action deliver = {
    nondet e = events.oneOf()
    all {
      evt' = evt.union(Set(e)),
      prog' = prog,
      chk'  = chk,
      led'  = led,
    }
  }

  action step = any {
    all { prog.length() > 0, stepHead },
    deliver,
  }

  // --- invariants ---

  // I5 (a slice of T3, as a reachability cross-check): the ledger never holds a
  // duplicate event-id. Holds because Ledger-Append fires only on a fresh eid.
  // Breakable: set `buggyLedger = true` and this fails on a reachable run.
  val ledgerNodup: bool =
    led.length() == led.foldl(Set(), (acc, x) => acc.union(Set(x))).size()

  // Expected to be violated: the workflow runs to its end, through the resume on event 5 and
  // both appends, so `ledgerNodup` is checked over states that reach them.
  val neverFinished: bool = prog.length() > 0

  // I2 (totality), v0 form: the workflow is never *stuck* — either it is done
  // (prog empty), or its head op can make progress, or its head is an Await still
  // waiting (which `deliver` can always unblock).
  //
  // SUPERSEDED, and it CANNOT FAIL: every arm below returns `true`, the `Await` arm literally
  // `or true`. `scripts/formal_checks.sh` does not run it, and running it would prove nothing.
  // `gather.qnt`'s `total` is the real check — it asks whether a head has a transition in the
  // position it actually landed in, which is where the suspend-in-gather hole lived. Kept as
  // the v0 form for the record, not as an assurance.
  val notStuck: bool =
    prog.length() == 0 or
    (match prog.head() {
      | Step(_)   => true
      | Ledger(_) => true
      | Await(e)  => evt.contains(e) or true   // park is legitimate, not stuck
    })
}
