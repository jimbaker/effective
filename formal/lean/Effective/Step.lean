/-!
# Effective — rules track (T2: determinism / invariant I6)

Formalizes the transition relation `κ → κ'` the handler defines, and the claim that — given the recorded
results — it is a *partial function* (at most one rule fires). This underwrites
replay: re-running the deterministic computation against a populated checkpoint map
reproduces the original outcome.

This is the §8.2 T2 nodes:

* **L2.1** — the `Op` datatype, the `State` fragment the rules read/write, and the
  `Step` relation (one constructor per rule).
* **L2.4 (`step_deterministic`)** — `Step` is the graph of a partial function.
* **L2.5 (`replay_stable`)** — once a checkpoint is recorded, the outcome is forced
  to replay it; the effect oracle is never consulted.

**The oracle ω.** Step-Run's premise `τ ⇓ v` is *effectful*; determinism can't hold
against arbitrary effects. Per the §2.3 precision note we model the effect as an
oracle `ω : Name → Val` supplied to the relation, so `run` binds `ω n` rather than
racing a side effect. T2 is determinism *for a fixed ω*; replay (L2.5) is the case
where ω is never read because `C` already holds the value.

**Scope.** We model the three *guard-bearing* ops — the ones with two rules whose
guards must be mutually exclusive, where determinism is non-trivial. `sleep` is a
trivial single-rule op (deterministic for lack of an alternative); `gather` is
compound and inherits determinism from the sub-relation, which is not modeled here.
-/

namespace Effective.Step

abbrev Name := Nat
abbrev Val := Nat
abbrev Event := Nat

/-- The fragment of a configuration the transition rules read and write: checkpoints
    `C` (disposable), the ledger `L` as the list of committed event-ids (append-only),
    and delivered events `E`. The workflow continuation `W` is threaded
    deterministically by the generator and is argued (not modeled here) to add no
    choice of its own. -/
structure State where
  C : Name → Option Val
  L : List Nat
  E : Event → Option Val

/-- The guard-bearing ops of §2.1. -/
inductive Op where
  | step   (n : Name)
  | await  (e : Event)
  | ledger (eid : Nat)

/-- What one step yields: resume the generator with a value and a new state; or
    commit (ledger ops bind nothing); or park (await with no event delivered yet). -/
inductive Outcome where
  | next   (v : Val) (σ : State)
  | commit (σ : State)
  | parked (e : Event)

/-- The transition relation `κ → κ'` (§2.3) on the guard-bearing ops, parameterized
    by the effect oracle `ω`. Each constructor is exactly one inference rule; the
    hypothesis on each is its guard. -/
inductive Step (ω : Name → Val) : Op → State → Outcome → Prop where
  | run    {n σ}    (h : σ.C n = none)   :
      Step ω (.step n) σ (.next (ω n) { σ with C := fun m => if m = n then some (ω n) else σ.C m })
  | replay {n v σ}  (h : σ.C n = some v) :
      Step ω (.step n) σ (.next v σ)
  | resume {e v σ}  (h : σ.E e = some v) :
      Step ω (.await e) σ (.next v σ)
  | park   {e σ}    (h : σ.E e = none)   :
      Step ω (.await e) σ (.parked e)
  | append {eid σ}  (h : eid ∉ σ.L)      :
      Step ω (.ledger eid) σ (.commit { σ with L := σ.L ++ [eid] })
  | idem   {eid σ}  (h : eid ∈ σ.L)      :
      Step ω (.ledger eid) σ (.commit σ)

/-- **T2 / L2.4 (determinism, invariant I6).** Given a fixed oracle `ω`, the
    transition relation is the graph of a partial function: for one op in one state,
    the outcome is unique — at most one rule fires. The off-diagonal cases close
    because each op's two guards are mutually exclusive (`none` vs `some`, `∉` vs
    `∈`); the diagonal cases close because each rule's output is a function of its
    inputs. -/
theorem step_deterministic (ω : Name → Val) {op σ o₁ o₂}
    (h₁ : Step ω op σ o₁) (h₂ : Step ω op σ o₂) : o₁ = o₂ := by
  cases h₁ <;> cases h₂ <;> simp_all

/-- **L2.5 (replay-stability).** Once the checkpoint for `n` is recorded as `v`, the
    step's outcome is forced to be `replay`ing `v` — the oracle `ω` is never
    consulted. This is precisely why a recorded run replays identically without
    re-running its effects. -/
theorem replay_stable (ω : Name → Val) {n σ v o}
    (hrec : σ.C n = some v) (h : Step ω (.step n) σ o) : o = .next v σ := by
  cases h with
  | run hnone => simp_all
  | replay hsome => simp_all

/-! ## Trust check -/

#check @step_deterministic
#print axioms step_deterministic
#check @replay_stable
#print axioms replay_stable

end Effective.Step
