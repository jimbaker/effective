import Mathlib.Data.Multiset.Bind
import Mathlib.Data.Multiset.Basic
import Mathlib.Data.List.Perm.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Logic.Relation

/-!
# Effective — budget accrual under `gather`: confluence as permutation-invariance

The operational question: when concurrent `gather`
branches accrue spend, is the observable a function of the **inputs** or of the
**schedule**? Replay is valid only if it is schedule-independent — i.e. **confluent**.

Two designs, formalized so the comparison is like-for-like at the point that matters:

* **Model B (per-branch subtotals, barrier fold)** — each branch accrues against its own
  budget. The *operational* content is `operational_run_eq_executedB`: an interleaved
  `Step` relation over a BAG of running branches (one op of any branch at a time — the real
  concurrency) reaches, on **every** draining schedule, the *same* executed bag
  `executedB`. Confluence then falls out (`operational_confluent`), and the algebraic
  `executedB_perm_invariant` records that the aggregate itself ignores branch order. The
  proof is a **conservation law**: `finalFrom` (current `exe` + every branch's remaining
  contribution) is invariant under each step — so no interleaving enumeration is needed and
  **no lock** is needed, because there is no shared cell to guard.
* **Model S (one shared meter, gated mid-fold)** — the gate reads the running partial sum,
  so the observable is a *left fold over an ordered list*, order-dependent
  (`shared_gate_order_dependent`). The model is already *atomic* (`ranFrom` reads-then-writes
  in one step), i.e. the meter is **locked**, and it is *still* non-confluent — a lock does
  not restore confluence.
-/

namespace Effective.Budget

open Relation

/-- An op: an identity and a cost. -/
abbrev Op := ℕ × ℕ

def ident (o : Op) : ℕ := o.1
def cost (o : Op) : ℕ := o.2

/-- The mid-fold **gate**: fold ops left-to-right threading accrued `spent`; an op RUNS iff
    the accumulated spend has not yet reached `limit`, and once it has, this op — and every
    op after it — is refused (spend is monotone). Returns the ids that ran, in order. -/
def ranFrom (limit : ℕ) : ℕ → List Op → List ℕ
  | _, [] => []
  | spent, o :: rest =>
    if spent < limit then ident o :: ranFrom limit (spent + cost o) rest
    else []

/-- Ids that ran under a per-branch meter, from a clean start. -/
def ranIds (limit : ℕ) (ops : List Op) : List ℕ := ranFrom limit 0 ops

/-! ## Model B — the aggregate, and its permutation-invariance (the algebraic side) -/

/-- **Model B aggregate.** The concurrent branches are a `Multiset`; each runs against its own
    budget `pb`; the executed-id bag is the multiset union. -/
def executedB (pb : ℕ) (branches : Multiset (List Op)) : Multiset ℕ :=
  branches.bind (fun b => (ranIds pb b : Multiset ℕ))

/-- The aggregate as a sum of per-branch coe-lists — the bridge used to connect the
    operational fold's conserved quantity to `executedB`. -/
theorem executedB_coe (pb : ℕ) (branches : List (List Op)) :
    executedB pb (branches : Multiset (List Op))
      = (branches.map (fun b => (ranIds pb b : Multiset ℕ))).sum := by
  simp [executedB, Multiset.bind, Multiset.map_coe, Multiset.sum_coe, Multiset.join]

/-- **Algebraic permutation-invariance of the aggregate.** Two branch orderings that are
    permutations coerce to the same `Multiset`, so `executedB` agrees. On its own this only
    says a function of a multiset ignores its input's list order; the *operational* weight is
    carried by `operational_run_eq_executedB` below, which proves the interleaved fold equals
    this aggregate. The two compose into confluence. -/
theorem executedB_perm_invariant (pb : ℕ) {l l' : List (List Op)} (h : l.Perm l') :
    executedB pb (l : Multiset (List Op)) = executedB pb (l' : Multiset (List Op)) :=
  congrArg (executedB pb) (Multiset.coe_eq_coe.mpr h)

/-- **The barrier sum is permutation-invariant** — a citation anchor for `List.Perm.sum_eq`
    (the commutative-monoid reason a per-branch fold needs no lock); no new content. -/
theorem barrier_sum_perm_invariant {l l' : List ℕ} (h : l.Perm l') : l.sum = l'.sum :=
  h.sum_eq

/-! ## Model B — the operational fold: the interleaved `Step` relation reaches the aggregate

This is the non-hollow core. `executedB` is the *aggregate*; here we model the *operational*
interleaved fold as a step relation and prove every draining run reaches that aggregate. -/

/-- A branch mid-run: its remaining ops and its own accrued spend — a per-branch cell touched
    by no other branch (the source of confluence). -/
abbrev BranchState := List Op × ℕ

/-- A configuration of the interleaved fold: a BAG of running branches (order among the
    branches forgotten — the concurrency) and the executed-id bag so far. -/
structure Config where
  bag : Multiset BranchState
  exe : Multiset ℕ

/-- One op-step of *some* branch in the bag — the interleaving the handler's threads realize.
    `run`: gate open (`spent < pb`), the head op runs (id joins `exe`; the branch advances and
    accrues). `halt`: gate shut, the branch is refused and leaves the bag (its remaining work
    drops — the park). `drained`: an empty branch leaves the bag. Which branch steps is the
    schedule. -/
inductive Step (pb : ℕ) : Config → Config → Prop
  | run {o rest spent bag exe} (h : spent < pb) :
      Step pb ⟨(o :: rest, spent) ::ₘ bag, exe⟩
             ⟨(rest, spent + cost o) ::ₘ bag, ident o ::ₘ exe⟩
  | halt {o rest spent bag exe} (h : ¬ spent < pb) :
      Step pb ⟨(o :: rest, spent) ::ₘ bag, exe⟩ ⟨bag, exe⟩
  | drained {spent bag exe} :
      Step pb ⟨([], spent) ::ₘ bag, exe⟩ ⟨bag, exe⟩

/-- What a branch state will still contribute if drained from here (its own gated fold). -/
def contrib (pb : ℕ) (bs : BranchState) : Multiset ℕ := (ranFrom pb bs.2 bs.1 : Multiset ℕ)

/-- The **conserved quantity**: what `exe` WILL be once the whole bag drains — the current
    `exe` plus every branch's remaining contribution. Each step moves ids from a branch's
    contribution into `exe` (or drops a shut-gate/empty branch whose contribution is empty),
    so this total never changes. -/
def finalFrom (pb : ℕ) (c : Config) : Multiset ℕ :=
  c.exe + (c.bag.map (contrib pb)).sum

/-- The starting configuration: every branch un-run with a zero meter, nothing executed. -/
def initConfig (branches : List (List Op)) : Config :=
  ⟨(branches.map (fun b => (b, 0)) : Multiset BranchState), 0⟩

/-- **Each step conserves `finalFrom`.** Three cases, each a local multiset computation:
    a `run` moves `ident o` from the branch's contribution into `exe`; a `halt` drops a branch
    whose contribution is already empty (gate shut ⇒ `ranFrom … = []`); a `drained` drops an
    empty branch (contribution `[]`). -/
theorem step_preserves_finalFrom (pb : ℕ) {c c'} (h : Step pb c c') :
    finalFrom pb c' = finalFrom pb c := by
  cases h with
  | @run o rest spent bag exe hgate =>
    have hr : ranFrom pb spent (o :: rest) = ident o :: ranFrom pb (spent + cost o) rest := by
      simp [ranFrom, hgate]
    simp only [finalFrom, contrib, Multiset.map_cons, Multiset.sum_cons, hr]
    rw [← Multiset.cons_coe]
    simp only [Multiset.cons_add, Multiset.add_cons]
  | @halt o rest spent bag exe hgate =>
    have hr : ranFrom pb spent (o :: rest) = [] := by simp [ranFrom, hgate]
    simp [finalFrom, contrib, Multiset.map_cons, Multiset.sum_cons, hr]
  | @drained spent bag exe =>
    simp [finalFrom, contrib, Multiset.map_cons, Multiset.sum_cons, ranFrom]

/-- **Operational confluence to the aggregate.** Every draining run of the interleaved fold —
    ANY schedule of per-branch op-steps from the start — reaches the *same* executed bag,
    `executedB pb ↑branches`. Proof: `finalFrom` is conserved along the run, equals
    `executedB` at the start, and equals `exe` at a drained (empty-bag) terminal. This is the
    ∀ the header claims — now earned operationally, not by the modeling choice alone. -/
theorem operational_run_eq_executedB (pb : ℕ) (branches : List (List Op)) {term : Config}
    (hrun : ReflTransGen (Step pb) (initConfig branches) term) (hterm : term.bag = 0) :
    term.exe = executedB pb (branches : Multiset (List Op)) := by
  have hconserve : finalFrom pb term = finalFrom pb (initConfig branches) := by
    clear hterm
    induction hrun with
    | refl => rfl
    | tail _ hlast ih => rw [step_preserves_finalFrom pb hlast]; exact ih
  have hterm_exe : finalFrom pb term = term.exe := by
    simp [finalFrom, hterm]
  have hinit : finalFrom pb (initConfig branches)
      = executedB pb (branches : Multiset (List Op)) := by
    simp only [finalFrom, initConfig, ranIds, executedB_coe, zero_add,
      Multiset.map_coe, Multiset.sum_coe, List.map_map]
    rfl
  rw [← hterm_exe, hconserve, hinit]

/-- Two draining runs from the same start agree — the confluence the replay path needs. -/
theorem operational_confluent (pb : ℕ) (branches : List (List Op)) {t t' : Config}
    (h : ReflTransGen (Step pb) (initConfig branches) t) (ht : t.bag = 0)
    (h' : ReflTransGen (Step pb) (initConfig branches) t') (ht' : t'.bag = 0) :
    t.exe = t'.exe := by
  rw [operational_run_eq_executedB pb branches h ht,
      operational_run_eq_executedB pb branches h' ht']

/-! ## Model S — the shared-meter gate is order-dependent (not a multiset function) -/

/-- **Non-confluence of Model S.** Two permutations of one branch-set whose shared-meter
    executed-id lists differ: `[a₁, a₂, b₁]` runs `{a₁,a₂}` then refuses `b₁`; `[a₁, b₁, a₂]`
    runs `{a₁,b₁}` then refuses `a₂` (limit 10, each cost 6). The observable depends on the
    schedule — replay-invalid. The model is atomic (locked), so this divergence survives a
    lock. -/
theorem shared_gate_order_dependent :
    ∃ l l' : List Op, l.Perm l' ∧ ranIds 10 l ≠ ranIds 10 l' := by
  refine ⟨[(10, 6), (11, 6), (20, 6)], [(10, 6), (20, 6), (11, 6)], ?_, ?_⟩
  · decide
  · decide

/-- **A lock does not restore confluence, stated axiom-free.** The two schedules are genuine
    permutations (proved by explicit constructors — no `Decidable`/`Quot`, so this stays
    axiom-free), yet the atomic (locked) shared meter runs different ids. For THIS
    uniform-cost witness the *total* happens to coincide (both run two ops of cost 6 before
    the trip), so a lock — which only makes the meter equal the actual spend — cannot even
    see the divergence. With heterogeneous costs the total diverges too (a strictly worse
    failure); the claim to keep is: a lock restores neither the trip nor, in general, the
    total. -/
theorem shared_lock_insufficient :
    let l : List Op := [(10, 6), (11, 6), (20, 6)]
    let l' : List Op := [(10, 6), (20, 6), (11, 6)]
    l.Perm l' ∧ (ranIds 10 l).length = (ranIds 10 l').length ∧ ranIds 10 l ≠ ranIds 10 l' := by
  refine ⟨(List.Perm.swap (20, 6) (11, 6) []).cons (10, 6), ?_, ?_⟩
  · rfl
  · decide

/-! ## Trust check -/

#check @operational_run_eq_executedB
#print axioms operational_run_eq_executedB
#print axioms operational_confluent
#print axioms executedB_perm_invariant
#print axioms shared_gate_order_dependent
#print axioms shared_lock_insufficient

end Effective.Budget
