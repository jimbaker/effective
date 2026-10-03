import Mathlib.Data.List.Permutation

/-!
# Effective: the `govern` gate's ruling `combine`

`govern` composes **policies** (budget, permission, quota) into one ruling per op:
`proceed | park | refuse`. This file models the fold that produces it, the third member of the
family `EnforceMeasured.lean` (budget's trip) and `Decide.lean` (permission's cascade) started.
With all three pure, unifying them is a refactor over proven references.

What is proved here is the *control shape*, and deliberately nothing about money or approval:

| property              | statement                                                                  |
|-----------------------|----------------------------------------------------------------------------|
| totality, determinism | by construction: a Lean function into a closed variant set                 |
| refuse dominates park | a gate never asks a human to grant past a policy that already refused; the |
|                       | fail-closed direction, with teeth in `combineBad_diverges`                 |
| no silent proceed     | if any policy objects, the ruling is not `proceed`                         |
| order-freedom         | permuting the policies never changes *which* ruling comes out              |
|                       | (`combine_kind_perm_invariant`); `serve`'s argument order is its semantics |
|                       | and `govern`'s is not                                                      |
| payload determinism   | the fused asks are the concatenation of the parking policies' asks, in     |
|                       | argument order, so a merged prompt reads in assembly order                 |

Classification is by **constructor**: a `refuse []` that forgot to say why still refuses, so an
empty list cannot silence a guard. The Python `combine` matches this, and the vectors pin the case.

Asks and reasons are abstract `ℕ` tags; their content is a value-axis detail discharged by
sampling in the Python conformance test, the same posture the sibling files take.

The park→resolve→resume→worker-death *protocol* around this ruling is Quint's half
(`formal/quint/govern_park.qnt`), which model-checks exactly-once delivery and the no-lost-grant
invariant this file's sequential ruling cannot see.
-/

namespace Effective.Govern

/-- One policy's answer about one op. `park` carries its share of the fused question; `refuse`
    carries its reasons. Mirrors `govern.Proceed | Park | Refuse`. -/
inductive Verdict
  | proceed
  | park (asks : List ℕ)
  | refuse (reasons : List ℕ)
  deriving DecidableEq, Repr

/-- The ruling's *identity*, payload erased — what "order-free" is a statement about. -/
inductive Kind
  | proceed
  | park
  | refuse
  deriving DecidableEq, Repr

def kind : Verdict → Kind
  | .proceed => .proceed
  | .park _ => .park
  | .refuse _ => .refuse

def isRefuse : Verdict → Bool
  | .refuse _ => true
  | _ => false

def isPark : Verdict → Bool
  | .park _ => true
  | _ => false

/-- Every refusing policy's reasons, concatenated in argument order. -/
def refusals : List Verdict → List ℕ
  | [] => []
  | .refuse rs :: vs => rs ++ refusals vs
  | _ :: vs => refusals vs

/-- Every parking policy's asks, concatenated in argument order — the FUSED question. -/
def asks : List Verdict → List ℕ
  | [] => []
  | .park a :: vs => a ++ asks vs
  | _ :: vs => asks vs

/-- The council's conjunctive fold: any refuse refuses, else any park parks, else proceed. -/
def combine (vs : List Verdict) : Verdict :=
  if vs.any isRefuse then .refuse (refusals vs)
  else if vs.any isPark then .park (asks vs)
  else .proceed

/-! ## Theorems -/

/-- **An empty council proceeds** — the fold's unit. (The DRIVER rejects a policy-less gate at
    assembly; that is an assembly rule, not an algebraic one.) -/
theorem empty_proceeds : combine [] = Verdict.proceed := rfl

/-- **Refuse dominates park.** If any policy refuses, the ruling refuses — however many policies
    wanted to park. A gate never asks a human to grant past a no. -/
theorem refuse_dominates (vs : List Verdict) (h : vs.any isRefuse = true) :
    kind (combine vs) = Kind.refuse := by
  simp [combine, h, kind]

/-- **Park only in the absence of a refusal**, and only when someone asked. -/
theorem park_when_asked_and_unrefused (vs : List Verdict)
    (hr : vs.any isRefuse = false) (hp : vs.any isPark = true) :
    kind (combine vs) = Kind.park := by
  simp [combine, hr, hp, kind]

/-- **No silent proceed.** If any policy objects — parks or refuses — the ruling is not `proceed`.
    Nothing in the fold turns an objection into permission. -/
theorem no_silent_proceed (vs : List Verdict) (v : Verdict) (hv : v ∈ vs)
    (hobj : v ≠ Verdict.proceed) : combine vs ≠ Verdict.proceed := by
  have hobjects : vs.any isRefuse = true ∨ vs.any isPark = true := by
    cases v with
    | proceed => exact absurd rfl hobj
    | park a => exact Or.inr (List.any_eq_true.mpr ⟨_, hv, rfl⟩)
    | refuse r => exact Or.inl (List.any_eq_true.mpr ⟨_, hv, rfl⟩)
  rcases hobjects with h | h
  · simp [combine, h]
  · by_cases hR : vs.any isRefuse = true <;> simp [combine, hR, h]

/-- **The fused payload is the concatenation of the parking policies' asks, in argument order** —
    so a merged prompt reads in the order the gate was assembled. -/
theorem park_asks_fuse (vs : List Verdict)
    (hr : vs.any isRefuse = false) (hp : vs.any isPark = true) :
    combine vs = Verdict.park (asks vs) := by
  simp [combine, hr, hp]

/-- **The ruling is order-free.** Permuting the policies cannot change WHICH ruling comes out.
    `govern` is a council of peers; `serve` is a queue whose order is its semantics — this
    theorem is that asymmetry, machine-checked. (The fused payload does reorder with the input;
    only the `kind` is invariant, which is exactly the claim.) -/
theorem combine_kind_perm_invariant {vs ws : List Verdict} (h : vs.Perm ws) :
    kind (combine vs) = kind (combine ws) := by
  have hr : vs.any isRefuse = ws.any isRefuse := by
    apply Bool.eq_iff_iff.mpr
    rw [List.any_eq_true, List.any_eq_true]
    exact ⟨fun ⟨x, hx, hp⟩ => ⟨x, h.mem_iff.mp hx, hp⟩, fun ⟨x, hx, hp⟩ => ⟨x, h.mem_iff.mpr hx, hp⟩⟩
  have hp : vs.any isPark = ws.any isPark := by
    apply Bool.eq_iff_iff.mpr
    rw [List.any_eq_true, List.any_eq_true]
    exact ⟨fun ⟨x, hx, hq⟩ => ⟨x, h.mem_iff.mp hx, hq⟩, fun ⟨x, hx, hq⟩ => ⟨x, h.mem_iff.mpr hx, hq⟩⟩
  by_cases hR : ws.any isRefuse = true
  · simp [combine, hr, hR, kind]
  · by_cases hP : ws.any isPark = true <;> simp [combine, hr, hp, hR, hP, kind]

/-- **A payload-less refusal still refuses.** Classification is by constructor: deciding on "did a
    reason arrive?" would let an under-populated verdict fall through to a park, i.e. a guard
    silenceable by an empty list. -/
theorem empty_refusal_still_refuses (a : List ℕ) :
    kind (combine [Verdict.refuse [], Verdict.park a]) = Kind.refuse := rfl

/-! ## Teeth (rung-1): the polarity, inverted -/

/-- The dangerous variant: **park dominates refuse** — the gate asks a human to grant past a
    policy that already said no, which is how a guard becomes a formality. -/
def combineBad (vs : List Verdict) : Verdict :=
  if vs.any isPark then .park (asks vs)
  else if vs.any isRefuse then .refuse (refusals vs)
  else .proceed

/-- **The inverted-polarity variant diverges** — `decide`-checked, axiom-free. -/
theorem combineBad_diverges : ∃ vs : List Verdict, combine vs ≠ combineBad vs := by
  refine ⟨[Verdict.refuse [1], Verdict.park [2]], ?_⟩
  decide

/-! ## Conformance vectors -/

/-- The canonical rows: one per fold arm, plus the dominance, order, and empty-payload cases. -/
def conformanceVectors : List (List Verdict × Verdict) :=
  [ ([], .proceed),
    ([.proceed, .proceed], .proceed),
    -- a single objection of each kind
    ([.park [1]], .park [1]),
    ([.refuse [1]], .refuse [1]),
    -- asks and reasons fuse, in argument order
    ([.park [1], .proceed, .park [2]], .park [1, 2]),
    ([.refuse [1], .refuse [2]], .refuse [1, 2]),
    ([.park [2], .park [1]], .park [2, 1]),
    -- refuse dominates park, in both orders (the RULING is order-free)
    ([.park [1], .refuse [2]], .refuse [2]),
    ([.refuse [2], .park [1]], .refuse [2]),
    ([.park [1], .refuse [2], .proceed], .refuse [2]),
    -- a refusal with no stated reason still refuses
    ([.refuse [], .park [1]], .refuse []),
    -- a park with no asks still parks (the same rule, other constructor)
    ([.park [], .proceed], .park []) ]

/-- **Every conformance vector holds against the model** — `decide`-checked, axiom-free. -/
theorem conformance_vectors_hold :
    ∀ vo ∈ conformanceVectors, combine vo.1 = vo.2 := by
  decide

/-! ## Trust check -/

#check @combine
#print axioms empty_proceeds
#print axioms refuse_dominates
#print axioms park_when_asked_and_unrefused
#print axioms no_silent_proceed
#print axioms park_asks_fuse
#print axioms combine_kind_perm_invariant
#print axioms empty_refusal_still_refuses
#print axioms combineBad_diverges
#print axioms conformance_vectors_hold

end Effective.Govern
