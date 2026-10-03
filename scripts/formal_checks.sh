# The REGISTERED formal checks: the single source of truth for both gates.
#
# Sourced by `scripts/formal_verify_container.sh` (the sandboxed default) and
# `scripts/formal_verify.sh` (the no-Podman host fallback), so the two run the same list.
#
# Each script supplies its own `run_check` / `run_expect_violation` / `run_tooth`; the LIST
# lives here, so a check added to one gate is in the other by construction.

# Checks that must PASS. One quint invocation each, argv-style.
FORMAL_CHECKS=(
    "verify effective.qnt --invariant=ledgerNodup --max-steps=14"
    "verify gather.qnt --invariant=total --max-steps=16"
    # noDeadlock on the explicit-state backend: the model is finite and tiny (54 states,
    # diameter 12), so TLC enumerates it COMPLETELY in ~4 s. The verdict is unbounded, and
    # strictly stronger than Apalache's bounded-14, which took ~170 s in the SMT encoding of
    # the invariant.
    "verify gather.qnt --invariant=noDeadlock --max-steps=14 --backend=tlc"
    "verify gather.qnt --temporal=liveness --max-steps=14 --backend=tlc"
    # Budget accrual: Model B (per-branch fold) is confluent operationally, the cross-check
    # of Lean's Effective.Budget.operational_run_eq_executedB (which proves it for ALL
    # schedules).
    "verify budget_confluence.qnt --invariant=confluent --max-steps=8"
    # The govern park protocol: park -> request ->
    # deliver -> resume -> worker-death, covering BOTH gates through the shared transition.
    "verify govern_park.qnt --invariant=safe --max-steps=12"
    "verify govern_park.qnt --temporal=liveness --max-steps=10 --backend=tlc"
    # Race and quorum: the choice, the flag and one global crash. TLC enumerates the
    # whole graph, so each verdict is unbounded.
    "verify race.qnt --invariant=safe --max-steps=40 --backend=tlc"
    "verify race.qnt --temporal=liveness --max-steps=40 --backend=tlc"
)

# Checks on the UNMODIFIED models that must VIOLATE.
#
# | kind                  | what the counterexample proves                                   |
# |-----------------------|------------------------------------------------------------------|
# | `nonVacuityWitness`   | a fair run terminates, so `fairness` is satisfiable and each     |
# |                       | `liveness` check above is not hollow                             |
# | `never<Kind>`         | the state the invariants quantify over is reachable              |
# | a strongest reading   | a property a document could be read to claim, and the run that   |
# |                       | breaks it                                                        |
FORMAL_EXPECT_VIOLATION=(
    "gather.qnt|--temporal=nonVacuityWitness --max-steps=14 --backend=tlc|gather fairness is satisfiable"
    "govern_park.qnt|--temporal=nonVacuityWitness --max-steps=10 --backend=tlc|govern fairness is satisfiable"
    "race.qnt|--temporal=nonVacuityWitness --max-steps=40 --backend=tlc|race fairness is satisfiable"
    # Each choice kind is reachable, so no invariant above holds by never seeing one.
    "race.qnt|--invariant=neverWinners --max-steps=40 --backend=tlc|a race can choose winners"
    "race.qnt|--invariant=neverImpossible --max-steps=40 --backend=tlc|a race can answer impossible"
    "race.qnt|--invariant=neverTimeout --max-steps=40 --backend=tlc|a race can time out"
    "race.qnt|--invariant=neverZeroReturns --max-steps=40 --backend=tlc|a race for zero returns"
    "race.qnt|--invariant=neverFullQuorumReturns --max-steps=40 --backend=tlc|a quorum of every branch returns"
    # The race's strongest reading, and what breaks it: a loser never reruns an op that a crash
    # interrupted after the choice, so the op has no result and its ledger row no checkpoint.
    "race.qnt|--invariant=everyAdmittedRecordedAtBarrier --max-steps=40 --backend=tlc|a crash loses a loser's admitted op"
    "race.qnt|--invariant=everyRowHasItsCheckpoint --max-steps=40 --backend=tlc|a loser's row keeps no checkpoint"
)

# TEETH: flip a model's design toggle to the KNOWN-BUGGY value and assert the checker finds
# the counterexample. A committed guard, so drift that breaks a counterexample fails loudly;
# each of these reproduces a bug the project actually hit or deliberately rejected.
# Format: file | sed-from | sed-to | args | what must break
FORMAL_TEETH=(
    # The shared-meter design MUST violate confluence: the machine-checked "the lock is not
    # enough". Lean's shared_gate_order_dependent is the kernel-checked sibling.
    "budget_confluence.qnt|sharedMeter: bool = false|sharedMeter: bool = true|--invariant=confluent --max-steps=8|Model S must violate confluence"
    # An in-process park name breaks the re-bind-by-name mechanism across worker death: a
    # revived worker awaits a name whose answer was already consumed (the stale-approval
    # failure).
    "govern_park.qnt|durableName: bool = true|durableName: bool = false|--invariant=awaitingMatchesPass --max-steps=12|an ephemeral park name must strand a resolution"
    # A handler with no rule for await-in-branch: the suspend-in-gather hole.
    "gather.qnt|handlerFixed: bool = true|handlerFixed: bool = false|--invariant=total --max-steps=16|the suspend-in-gather hole must break totality"
    "gather.qnt|handlerFixed: bool = true|handlerFixed: bool = false|--invariant=noDeadlock --max-steps=14 --backend=tlc|the suspend-in-gather hole must deadlock"
    "gather.qnt|handlerFixed: bool = true|handlerFixed: bool = false|--temporal=liveness --max-steps=14 --backend=tlc|the suspend-in-gather hole must stall even under fairness"
    # Dropping the ledger append guard must produce a duplicate event id.
    "effective.qnt|buggyLedger: bool = false|buggyLedger: bool = true|--invariant=ledgerNodup --max-steps=14|an unguarded append must duplicate an event id"
    # The race's rejected designs. A flag published before the choice is saved stops a branch
    # that a re-run after a crash then records as the winner.
    "race.qnt|flagBeforeChoice: bool = false|flagBeforeChoice: bool = true|--invariant=everyStopHasItsChoice --max-steps=40 --backend=tlc|a flag before the choice must stop a loser for an unsaved choice"
    "race.qnt|flagBeforeChoice: bool = false|flagBeforeChoice: bool = true|--invariant=noWinnerWasStopped --max-steps=40 --backend=tlc|a flag before the choice must let a crash crown a stopped branch"
    "race.qnt|lateFlagCheck: bool = false|lateFlagCheck: bool = true|--invariant=noAdmitAfterFlag --max-steps=40 --backend=tlc|a loser that skips the flag must admit under it"
    "race.qnt|abandonAdmitted: bool = false|abandonAdmitted: bool = true|--invariant=admittedRecordedAtBarrier --max-steps=40 --backend=tlc|a loser dropping its admitted op must leave it unrecorded"
    "race.qnt|deadlineFromRestart: bool = false|deadlineFromRestart: bool = true|--invariant=deadlineNotExtended --max-steps=40 --backend=tlc|a deadline taken at restart must move later"
    "race.qnt|innerIgnoresFlag: bool = false|innerIgnoresFlag: bool = true|--invariant=noInnerChoiceAfterOuterFlag --max-steps=40 --backend=tlc|an inner race deaf to the flag must choose under it"
    "race.qnt|recoveryDrivesBeforeChoice: bool = false|recoveryDrivesBeforeChoice: bool = true|--invariant=noAdmitAfterFlag --max-steps=40 --backend=tlc|recovery driving a loser before the flag must admit under a published flag"
    "race.qnt|choiceInMemory: bool = false|choiceInMemory: bool = true|--invariant=choiceStable --max-steps=40 --backend=tlc|a choice held in memory must change across a crash"
    "race.qnt|impossibleFromOutcomes: bool = false|impossibleFromOutcomes: bool = true|--invariant=impossibleOnlyWhenHopeless --max-steps=40 --backend=tlc|impossible from unseen outcomes must answer while hope remains"
    "race.qnt|tieIsInTime: bool = false|tieIsInTime: bool = true|--invariant=winnersBeforeDeadline --max-steps=40 --backend=tlc|a completion at the deadline must not win"
    "race.qnt|returnBeforeQuiescent: bool = false|returnBeforeQuiescent: bool = true|--invariant=returnsQuiescent --max-steps=40 --backend=tlc|a return before quiescence must leave a loser live"
    "race.qnt|zeroStartsBranches: bool = false|zeroStartsBranches: bool = true|--invariant=zeroStartsNothing --max-steps=40 --backend=tlc|a race for zero that starts a branch must admit an op"
    "race.qnt|earlyTimeout: bool = false|earlyTimeout: bool = true|--invariant=timeoutOnlyAtDeadline --max-steps=40 --backend=tlc|a timeout before the deadline must be saved early"
)
