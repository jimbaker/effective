# Effective — Quint model (the reachability / interleaving half)

The model-checking side of the formal-verification investigation. **Lean** (`../lean/`)
proves the *pure* facts — name injectivity (T1), the transition relation is a partial
function (T2), the ledger is append-only (T3). **Quint** explores what Lean punts on:
the workflow continuation `W` and the reachable state space across interleavings, and
*imports* the Lean facts as modeling choices (a deterministic step; abstract names)
rather than re-deriving them at every state.
Lean and Quint together cover *the model* (the algebra + the reachable dynamics); the
model↔code correspondence is held by tests and adversarial agent/HITL review, which
"Out of scope" below explains.

## Status

**v0 — `effective.qnt`** (the harness): one straight-line workflow, the three
guard-bearing ops, one invariant *with teeth*.

| Invariant | What | Apalache (14 steps) |
|---|---|---|
| `ledgerNodup` | the ledger never holds a duplicate event-id (reachability cross-check of T3 / Ledger-Idem) | ✅ `NoError` |
| `notStuck` | totality, v0 form | ⚠️ **superseded and vacuous — not run.** Every match arm returns `true` (the `Await` arm is literally `or true`), so it has no failure mode, and `scripts/formal_checks.sh` does not invoke it. `total` in `gather.qnt` is the check that replaced it and can go red. |

Teeth: flip `buggyLedger = true` (drop the append guard) → the checker finds a
reachable `Ledger(10) → deliver(5) → resume → Ledger(10) ⇒ [10,10]` trace.

**v1 — `gather.qnt`** (where the model checker earns its keep): models `gather` —
branches running concurrently under their own continuations, the thing Lean's `Step`
relation punted on — and checks **I2 totality** faithfully, reproducing the
*suspend-in-gather* bug.

`total` encodes I2 directly: every reachable configuration's active
heads — main + each branch — have a *defined transition or park* (`headHandled`). It
is **not** a deadlock/liveness check — an await on an absent event is legitimately
Parked (I2-OK); whether it ever resumes is I3, a later rung.

| Property | What | Checked |
|---|---|---|
| `total` (I2 totality) | every reachable op-head has a transition-or-park (no suspend-in-gather hole) | ✅ Apalache `NoError` @20 (fixed) |
| `noDeadlock` | whole-state deadlock-freedom: every non-terminal config can progress or be unblocked by a resumable delivery | ✅ Apalache `NoError` @14 (fixed; ~5min — SMT-heavy) |
| `ledgerNodup` | the ledger never holds a duplicate (T3 cross-check) | ✅ Apalache `NoError` @20 |
| `liveness` (I3) | `fairness ⟹ eventually terminated` — a parked await resumes and the workflow completes | ✅ TLC, complete state space (fixed) |

`noDeadlock` is the per-*state* sibling of the per-*head* `total` — it catches a stuck
configuration even when it isn't a missing-rule one. Teeth: `handlerFixed=false` → it
fires at the in-branch `Await(7)` (`~3s`).

**I3 (liveness) is temporal**, so it's checked with TLC (`--backend=tlc`; Apalache's
temporal support is experimental + interactive). The result is three-sided and honest:
*without* fairness `eventuallyTerminates` **fails** (a stuttering counterexample — liveness
is conditional on fairness); *with* fairness the **fixed** handler holds (TLC checks the
whole state space — finite by construction: 54 distinct states, depth 12, identical at
`--max-steps` 11/16, so not bound-luck; the join *shrank* the space from the pre-join 106
by stopping main from interleaving past `Gather`); the **buggy** handler **fails** even with
fairness (the in-branch await is never enabled, so fairness can't force it — a permanent
stall). Liveness pins a failure mode distinct from the I2 hole, via a different tool.

Two honest scoping notes: (1) **non-vacuity is guarded** — `nonVacuityWitness`
(`fairness ⟹ always ¬terminated`) is checked to *expect a violation*; the counterexample
is a fair, terminating run, proving `fairness` is satisfiable (so `liveness` isn't
hollow). (2) This checks I3 on the **suspend-in-gather** instance; a lease-reclaim bug
is a *separate* I3 instance not modeled here. And the
unmodeled gather *join* (below) is harmless for *whether* the workflow terminates, though
it does permit `Ledger(30)` to commit before the branch ledgers (a §2.4 ordering
infidelity, Lean's via T2, not the liveness question).

Teeth — the real bug, machine-found: flip `handlerFixed = false` (a
handler with no rule for await-in-branch) → Apalache finds the reachable config
`Step(1) → Gather` (spawn) `→ branch0:Step(2) → branch0 head = Await(7)` that the
handler can't handle (`total` fails) in ~6s. The non-vacuous content is this *buggy*
direction (the hole is **reachable**); the fixed handler is total **by construction**
(every op-position has a rule), which `total` reflects. The result is **bounded**
(Apalache to N steps), but not bound-sensitive — `total` holds at 16/20/24/30 and the
violation sits at depth ~4; the reachable *shapes* are exhausted well within the bound
because continuations only shrink and `chk`/`evt`/`led` grow over a tiny finite
alphabet. The bug is caught for the *whole* (bounded) reachable space.

**v2 — `budget_confluence.qnt`** (accrual under gather; the operational half of a
Lean-led result): budget spend accrued across concurrent branches must be **confluent** —
schedule-independent — or replay is invalid. Confluence *is* permutation-invariance, and
here Lean is the state-space reducer: `formal/lean/Effective/Budget.lean` models the
branches as a **`Multiset`** and proves Model-B (per-branch fold) accrual permutation-invariant
for **all** inputs (`executedB_perm_invariant`) — no interleaving enumeration. This model is
the *operational* cross-check: `confluent` holds on every interleaving for Model B; flip
`sharedMeter = true` and Apalache returns the `[a₁,a₂,b₁]` schedule reaching `executed
= {10,11} ≠ {10,20}` — the operational witness of Lean's `shared_gate_order_dependent`. The
model is *atomic* (a step reads-then-writes `spent` in one action = a **locked** meter) and
still non-confluent: the machine-checked "the lock is not enough."

**v3 — `govern_park.qnt`** (the park protocol; the half `Govern.lean` disclaims): a
`govern` gate's suspend/resume cycle — park → request → deliver → resume → **worker death** →
fresh-worker replay — model-checked over every interleaving. Both gates' concurrency stories are model-checked here,
including a permission's survival across worker death. **One model covers both gates**, which is the
payoff of unifying them behind `govern`: a grant and an approval differ only in what the answer
*means*, and meaning is the policy's business, not the protocol's.

| Invariant | What | Checked |
|---|---|---|
| `failClosed` | a crash between park and answer never yields `proceeded` | ✅ Apalache @12 |
| `exactlyOnce` | each pass's resolution is folded in exactly once, in order (`absorbed = [0..n-1]`) — a re-delivery or a re-run cannot double-count a grant | ✅ Apalache @12 |
| `awaitingMatchesPass` | a parked worker awaits exactly the name its durable position implies — the no-call/cc claim itself | ✅ Apalache @12 |
| `bounded` | the gate settles within `maxPasses` rather than parking forever | ✅ Apalache @12 |
| `liveness` (I3) | under fairness (crashes eventually stop; enabled actions fire) the gate settles | ✅ TLC, non-vacuity guarded |

`safe` bundles the four safety invariants for one registered run. Fairness deliberately
excludes `die`: a crash is possible, never obligatory — and `eventually(always(alive))` is the
crash-stop assumption, since an adversary who kills the worker forever starves any protocol.

Teeth — and this one reproduces a bug the project actually hit: flip `durableName = false` so
the park name comes from an **in-process counter** instead of `|absorbed|`. It survives a clean
run and breaks only across worker death; Apalache returns `absorbed: [0], awaiting: 0` — a
revived worker awaiting a name whose answer was already consumed. That is the stale-approval
failure `permission.human`'s run-scoping docstring records (a suite's second run never parked,
resuming instantly on a persisted approval), caught by a model. The registered gate asserts the violation, so drift that breaks the counterexample
fails loudly.

**v4: `race.qnt`** (race and quorum): three branches race for `k` winners, with
`k` chosen at init from 0 to 3. An op is admitted, commits its ledger row, then its checkpoint.
The parent saves the choice and then publishes a flag naming the losers, and a flagged loser
stops at its next admission. One global crash loses every ephemeral bit, and recovery republishes
the flag from the saved choice before driving a branch. Each invariant checks one rule of the race,
read at the moment it can fail, and TLC enumerates the whole graph (11.3M distinct states,
2026-09-18; `just formal-verify` recomputes it).

| Invariant | the rule it checks |
|---|---|
| `choiceStable`, `everyStopHasItsChoice`, `noWinnerWasStopped` | once saved, the choice never changes; a loser stops only for the choice on record, and no winner is stopped |
| `exactlyK`, `zeroStartsNothing` | exactly `k` winners, each a completed success; `k = 0` starts nothing |
| `impossibleOnlyWhenHopeless` | the race is impossible only when, at the decision, successes plus unresolved branches fell short of `k` |
| `noAdmitAfterFlag`, over every incarnation and after a crash that found the choice saved; `admittedRecordedAtBarrier` | no loser admits an op once its flag is published; every op the returning incarnation admitted has its checkpoint |
| `returnsQuiescent` | the race returns only when no loser has an op in flight or can still admit one |
| `loserSpendReachesMeter` | every checkpointed op, a loser's included, is folded into the meter at the barrier |
| `noInnerChoiceAfterOuterFlag` | an inner race never saves a choice once its enclosing loser's flag is published |
| `deadlineNotExtended`, `winnersBeforeDeadline`, `timeoutOnlyAtDeadline` | a restart never moves the deadline, a winner completes before it, and a timeout comes at or after it |
| `liveness` | the race returns under weak fairness |

Two stronger readings are registered as violations: every op admitted in any incarnation has its
checkpoint at the barrier (`everyAdmittedRecordedAtBarrier`), and every ledger row has its
checkpoint (`everyRowHasItsCheckpoint`, the orphan row). A crash after the choice, with a loser's
op in flight, breaks both, and neither breaks without a crash. Each teeth toggle is a rejected
design.

## Run

Toolchain: **pinned per-machine via `just formal-setup`** (idempotent; nothing global) —
see `infra/formal/PIN.txt` for the co-versioning record and the supply-chain procedure.
Quint 0.32.0 installs into `infra/formal/node_modules` by `npm ci --ignore-scripts`
(lockfile sha512 integrity); `verify` auto-fetches **Apalache 0.56.1** (symbolic
checker → SMT/Z3; needs a JDK ≥ 17) into `~/.quint/`, and both `formal-setup` and
`formal-verify` refuse to run it unless `apalache.jar` matches the PIN's sha256
(recorded from two independent fetches on two machines, 2026-06-20 and 2026-07-04).

```sh
just formal-setup    # install + hash-verify the pinned toolchain on this machine
just formal          # fast gate: lake build + quint typecheck (seconds)
just formal-image    # build the sandbox image once per device (hash-gated build)
just formal-verify   # every registered check (~8 min) — sandboxed by default:
                     #   --network=none, read-only /spec, tmpfs workdir
just formal-verify-host   # bare-metal fallback (no Podman; run-time hash guard)
```

**The registered checks live in ONE place — `scripts/formal_checks.sh`** — which both the
sandboxed gate and the host fallback source. Read that file for the current list; do not
re-list them here, because a second list is a list that drifts: two hand-lists let the host
fallback run a subset and still print a green line a reader takes as equivalent to the
sandboxed run.

Three kinds of entry, all enforced:

| kind | must |
|---|---|
| `FORMAL_CHECKS` | pass |
| `FORMAL_EXPECT_VIOLATION` | **fail**: a `nonVacuityWitness` proving `fairness` satisfiable, a `never<Kind>` proving a state reachable, or a strongest reading of a document and the run that breaks it |
| `FORMAL_TEETH` | **fail**: a design toggle flipped to its known-buggy value must still produce its counterexample |

Every guard and tooth is registered there, so each one runs; together they cost ~27 s.

(`quint run effective.qnt --invariant=ledgerNodup --max-steps=12` — the random
simulator — needs the rust-evaluator, fetched on first `run`; deliberately unpinned
until a gate uses it, per PIN.txt.)

`/_apalache-out` (run output) is gitignored.

## Imported Lean assumptions

`gather.qnt` names the pure facts it takes as given from Lean rather than re-deriving
them (`assume T2_determinism`, `assume T3_appendOnly`, `assume L15_gatherNamingInjective`),
each citing its Lean theorem + axiom footprint and stating how the model relies on it.
This makes the division of labor auditable: Quint explores reachability; Lean owns the
algebra. (`ledgerNodup` is the model-side reflection of T3.)

## Next (the ladder, climbing)

- **Nested gather:** `branchSpecs` containing `Gather` (the `Gather => all { false }`
  branch arm becomes reachable) — exercises the cross-gather keying L1.4/L1.5 owns.
- **Nested gather**, continued: with the join modeled, a branch that itself contains
  `Gather` becomes the real test of cross-gather keying (and needs per-branch/strong
  fairness for liveness — aggregate weakFair only sufficed because branches were finite
  and non-nested).
- *(done)* ~~the gather **join**~~ — `stepMain` is gated while a gather is active and
  `joinGather` fires once branches drain, so main's continuation waits (§2.4). The only
  remaining §2.4 simplification is the **completion-order** (not index-order) ledger,
  kept deliberately — ordering/replay-stability is Lean's via T2.

## Out of scope: a mechanical model↔code conformance harness

A harness replaying spec-generated traces against the real handler (Quint Connect, or
a Python analogue) is **out of scope**. Closing the model→code gap *formally* is a
regress — you'd formalize the implementation, then justify *that* against the code, and
so on (not turtles all the way down, but we have turtles). For a formalization serving
a real product, that correspondence is held — appropriately — by the existing
recording/replay + crash-at-every-op tests *and* adversarial agent/HITL review
by **agency** rather than a formal harness.
