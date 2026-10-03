import Effective.Step
import Mathlib.Data.List.Basic
import Mathlib.Logic.Relation

/-!
# Effective — rules track (T3: append-only ledger / invariant I5)

Formalizes invariant **I5** of Effective's operational semantics on the `Step`
relation of `Effective.Step`: the ledger `L` is **append-only** — it grows
monotonically in prefix order along any run, and re-committing an already-recorded
event-id is a no-op (idempotent by `eid`).

The §8.2 T3 nodes:

* **L3.2 (`advance_ledger_mono`)** — one forward step extends `L` as a prefix.
* **L3.3 (`ledger_monotone`)** — the reflexive-transitive closure preserves the
  prefix order: `σ →* σ'` implies `σ.L <+: σ'.L`.
* **L3.4 (`ledger_idem`, `advance_nodup`)** — re-committing a recorded `eid` is a
  no-op; the ledger stays duplicate-free along a run.
* **L3.5 (`L_not_function_of_C`, `C_not_function_of_L`)** — the *modest* direction of
  non-derivability: in the model, neither bookkeeper is a function of the other. The
  *architectural* invariant (append-only enforced by a DB trigger; the inspect path
  cannot reach the ledger) is outside this model.
-/

namespace Effective.Ledger

open Effective.Step

/-- The continuing post-state of an outcome. `parked` halts, so it has none. -/
def contState : Outcome → Option State
  | .next _ σ => some σ
  | .commit σ => some σ
  | .parked _ => none

/-- One forward move of the configuration: some op steps in `σ`, and its outcome
    continues into `σ'`. -/
def Advances (ω : Name → Val) (σ σ' : State) : Prop :=
  ∃ op o, Step ω op σ o ∧ contState o = some σ'

/-- **L3.2 (per-step monotonicity).** Every forward move extends the ledger: `σ.L`
    is a prefix of `σ'.L`. Every rule but `append` leaves `L` untouched
    (prefix-refl); `append` grows it by appending (prefix-append). -/
theorem advance_ledger_mono (ω : Name → Val) {σ σ'} (h : Advances ω σ σ') :
    σ.L <+: σ'.L := by
  obtain ⟨op, o, hstep, hcont⟩ := h
  cases hstep with
  | run h    => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact List.prefix_refl _
  | replay h => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact List.prefix_refl _
  | resume h => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact List.prefix_refl _
  | park h   => simp [contState] at hcont
  | append h => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact List.prefix_append _ _
  | idem h   => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact List.prefix_refl _

/-- **T3 / L3.3 (append-only).** Along any run `σ →* σ'` (the reflexive-transitive
    closure of `Advances`), the ledger only grows: `σ.L` is a prefix of `σ'.L`.
    Induction on the closure, using transitivity of the prefix order and L3.2. -/
theorem ledger_monotone (ω : Name → Val) {σ σ'}
    (h : Relation.ReflTransGen (Advances ω) σ σ') : σ.L <+: σ'.L := by
  induction h with
  | refl => exact List.prefix_refl _
  | tail _ hlast ih => exact ih.trans (advance_ledger_mono ω hlast)

/-- **L3.4 (idempotent by `eid`).** If `eid` is already recorded, committing it again
    is a no-op — the outcome is `commit σ`, with `L` unchanged. The `append` rule's
    guard (`eid ∉ L`) is what forbids the duplicate; only `idem` can fire. -/
theorem ledger_idem (ω : Name → Val) {eid σ o}
    (hin : eid ∈ σ.L) (h : Step ω (.ledger eid) σ o) : o = .commit σ := by
  cases h with
  | append hnin => exact absurd hin hnin
  | idem _      => rfl

/-- **L3.4 (no duplicates).** A forward step preserves duplicate-freedom of the
    ledger: `append` only fires when `eid ∉ L`, so the new entry is fresh. -/
theorem advance_nodup (ω : Name → Val) {σ σ'}
    (hnd : σ.L.Nodup) (h : Advances ω σ σ') : σ'.L.Nodup := by
  obtain ⟨op, o, hstep, hcont⟩ := h
  cases hstep with
  | run h    => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact hnd
  | replay h => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact hnd
  | resume h => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact hnd
  | park h   => simp [contState] at hcont
  | append h => simp only [contState, Option.some.injEq] at hcont; subst hcont
                rw [List.nodup_append]
                refine ⟨hnd, by simp, ?_⟩
                intro a ha b hb
                simp only [List.mem_singleton] at hb
                subst hb
                exact fun heq => h (heq ▸ ha)
  | idem h   => simp only [contState, Option.some.injEq] at hcont; subst hcont; exact hnd

/-- **L3.5 (non-derivability — the modest, type-level direction).** The ledger is not
    a function of the checkpoints: two states share `C` yet differ in `L`. So `L`
    cannot be computed from `C`. -/
theorem L_not_function_of_C :
    ∃ σ₁ σ₂ : State, σ₁.C = σ₂.C ∧ σ₁.L ≠ σ₂.L :=
  ⟨⟨fun _ => none, [], fun _ => none⟩, ⟨fun _ => none, [0], fun _ => none⟩, rfl, by simp⟩

/-- **L3.5 (the converse direction).** Nor is `C` a function of `L`: two states share
    `L` yet differ in `C`. The two bookkeepers are independent in the model. -/
theorem C_not_function_of_L :
    ∃ σ₁ σ₂ : State, σ₁.L = σ₂.L ∧ σ₁.C ≠ σ₂.C :=
  ⟨⟨fun _ => none, [], fun _ => none⟩,
   ⟨fun n => if n = 0 then some 0 else none, [], fun _ => none⟩,
   rfl, by intro h; have := congrFun h 0; simp at this⟩

/-! ## Trust check -/

#check @ledger_monotone
#print axioms ledger_monotone
#check @ledger_idem
#print axioms ledger_idem
#print axioms advance_nodup
#print axioms L_not_function_of_C
#print axioms C_not_function_of_L

end Effective.Ledger
