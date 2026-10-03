import Mathlib.Data.Nat.Basic

/-!
# Effective — the measured trip transition `enforce_measured`, formalized once

The question this file answers: the measured-spend trip (`effective/budget.py::enforce_measured`,
mirrored by `DurableHandler._enforce_measured`) classifies a run at a metered `AskLLM` into exactly
one of three outcomes — **Cleared / Parked / Refused** — consulting a sequence of grants keyed by a
*trip index*. The transition is formalized **once**, here, and every Python interpreter of it is
held to the same machine-derived rows, so no interpreter re-opens the model/impl gap on its own.

Scope split: this file owns the *pure-axis* facts of the **sequential** trip — it
is a partial function of `(limit, meter, on_exhaust, grants)`, `on_exhaust="fail"` refuses **before**
consulting grants, the awaited grant's trip index is a deterministic
function of the inputs (so the durable park name `budget-grant:{run_id}:{trip}` re-derives on replay),
and a positive grant advances the trip deterministically. The *concurrent* accrual under `gather` is a
separate concern already proved confluent in `Budget.lean` (`operational_run_eq_executedB`); the trip
is sequential-only (a branch handler holds no budget), so there is no interleaving to enumerate here.

Money is modeled as `ℕ` in abstract units (the Python uses dollars; the semantics are threshold +
order, so `ℕ` suffices and keeps every check `decide`-able). The unit maps to the Python conformance
vectors by a fixed scale — see `Effective/enforce_vectors.json` and the conformance test that consumes
it. `(A-money)`: the reals→ℕ abstraction is discharged by *sampling* (the exported vectors are run
against the live float `enforce_measured`), not proved in Lean — a determinism-only, value-axis
boundary that gets no Lean node (the skill's tier rule).
-/

namespace Effective.EnforceMeasured

/-- A delivered grant: `stop` (refuse), or `add n` more units (`n = 0` is a zero grant — also a
    refusal, matching `grant.stop or grant.add_dollars <= 0.0`). -/
inductive Grant
  | stop
  | add (n : ℕ)
  deriving DecidableEq, Repr

/-- The trip classification — one constructor per outcome, the closed variant set a `match` must
    exhaust. `parked` carries the *trip index*; the durable park name is `budget-grant:{run_id}:{trip}`,
    so this index IS the park's identity (run_id fixed per run). -/
inductive Outcome
  | cleared (granted trips : ℕ)
  | parked (trip : ℕ)
  | refused (spent ceiling : ℕ)
  deriving DecidableEq, Repr

/-- The measured trip as ONE total function, structurally recursive on the grant list
    (so totality is by definition: the loop cannot diverge). Mirrors `enforce_measured`:
    while spend is at/over `limit + granted`, **fail first** (`on_exhaust="fail"`), else consult the
    grant for the current trip — absent ⇒ `parked`, `stop`/zero ⇒ `refused`, positive ⇒ refill and
    re-check with the trip advanced. `granted`/`trips` thread the accrual; the top-level entry seeds
    them at 0. -/
def enforce (limit meter : ℕ) (fail : Bool) : ℕ → ℕ → List Grant → Outcome
  | granted, trips, gs =>
    if meter < limit + granted then Outcome.cleared granted trips
    else
      match fail, gs with
      | true, _ => Outcome.refused meter (limit + granted)
      | false, [] => Outcome.parked trips
      | false, Grant.stop :: _ => Outcome.refused meter (limit + granted)
      | false, Grant.add 0 :: _ => Outcome.refused meter (limit + granted)
      | false, Grant.add (n + 1) :: rest =>
        enforce limit meter fail (granted + (n + 1)) (trips + 1) rest

/-- The trip from a clean start (`granted = 0`, `trips = 0`) — the interpreter's entry point. -/
def enforceMeasured (limit meter : ℕ) (fail : Bool) (gs : List Grant) : Outcome :=
  enforce limit meter fail 0 0 gs

/-! ## Theorems — the facts every interpreter of the trip must satisfy -/

/-- **Under the ceiling clears immediately** — no trip, no grant consulted (`granted = trips = 0`).
    The `if` guard fires before any grant is read. -/
theorem under_ceiling_clears (limit meter : ℕ) (fail : Bool) (gs : List Grant)
    (h : meter < limit) : enforceMeasured limit meter fail gs = Outcome.cleared 0 0 := by
  unfold enforceMeasured enforce
  split
  · rfl
  · omega  -- the else branch needs ¬ meter < limit + 0, contradicting h

/-- **fail-first.** With `on_exhaust="fail"`, an over-ceiling run refuses
    at the ceiling **before** consulting any grant — the outcome is `refused meter limit`, whatever
    the grants are. The fail arm is unconditional; `enforceBad` below models consulting grants first. -/
theorem fail_refuses_before_grants (limit meter : ℕ) (gs : List Grant)
    (h : limit ≤ meter) : enforceMeasured limit meter true gs = Outcome.refused meter limit := by
  have hlt : ¬ meter < limit := by omega
  simp [enforceMeasured, enforce, hlt]

/-- **The base park.** Over the ceiling, park-mode, no grant delivered ⇒ `parked 0`: the run awaits
    `budget-grant:{run_id}:0`, the deterministic first-trip name. -/
theorem park_when_no_grant (limit meter : ℕ)
    (h : limit ≤ meter) : enforceMeasured limit meter false [] = Outcome.parked 0 := by
  have hlt : ¬ meter < limit := by omega
  simp [enforceMeasured, enforce, hlt]

/-- **A stop grant refuses.** Over the ceiling, park-mode, the delivered grant is `stop` ⇒ refused
    (a `stop`/zero grant would only re-trip forever, so it is a refusal, not a re-park). -/
theorem stop_refuses (limit meter : ℕ) (rest : List Grant)
    (h : limit ≤ meter) :
    enforceMeasured limit meter false (Grant.stop :: rest) = Outcome.refused meter limit := by
  have hlt : ¬ meter < limit := by omega
  simp [enforceMeasured, enforce, hlt]

/-- **A zero grant refuses** (`add 0` = `add_dollars <= 0.0`). -/
theorem zero_grant_refuses (limit meter : ℕ) (rest : List Grant)
    (h : limit ≤ meter) :
    enforceMeasured limit meter false (Grant.add 0 :: rest) = Outcome.refused meter limit := by
  have hlt : ¬ meter < limit := by omega
  simp [enforceMeasured, enforce, hlt]

/-- **A positive grant advances the trip deterministically** (grant order / placement). Over the
    ceiling with a positive grant `add (n+1)` at trip `t`, the transition refills by `n+1` and
    re-checks at trip `t+1` on the remaining grants — so the trip index is a deterministic function
    of the consumed positive-grant prefix, which is what makes each awaited grant name
    `budget-grant:{run_id}:{t}` re-derive on replay. -/
theorem positive_grant_advances (limit meter granted trip n : ℕ) (rest : List Grant)
    (h : ¬ meter < limit + granted) :
    enforce limit meter false granted trip (Grant.add (n + 1) :: rest)
      = enforce limit meter false (granted + (n + 1)) (trip + 1) rest := by
  simp [enforce, h]

/-! ## The RUN-level overshoot bound

Everything above is one trip. The property an operator cares about is about the whole run:
*how far past the ceiling can spend get?* A pytest pin samples it; this section proves it.

The model: a run is a list of call costs, each admitted only while spend is strictly under the
ceiling (that IS the trip's `meter < limit + granted` guard), and the run stops at the first call
the guard rejects. Spend can overshoot by *one* call, and only by as much as that call cost:

    spend ≤ ceiling + c_max        with `ceiling := limit + Σ grants`

Overshoot is unavoidable: the meter can only be read *before* a call, and a call's true cost is
known only *after* it. The excess is bounded by a single call's cost and does not accumulate, so a
ceiling plus a known worst-case call price is a guarantee, and it is what a grantor decides against
at a park. -/

/-- One run: fold call `costs` into `spent`, admitting a call only while under `ceiling`, and
    stopping at the first call the gate rejects (structurally recursive, so it cannot diverge). -/
def runSpend (ceiling : ℕ) : ℕ → List ℕ → ℕ
  | spent, [] => spent
  | spent, c :: cs => if spent < ceiling then runSpend ceiling (spent + c) cs else spent

/-- **The run-level overshoot bound.** If every call costs at most `cmax` and the run starts
    within `ceiling + cmax`, it ends within `ceiling + cmax` — the excess never accumulates
    across calls, however many the run makes.

    The induction is on the call list, and the step is the whole argument: a call is admitted only
    when `spent < ceiling`, so the post-call spend is at most `(ceiling - 1) + cmax`. -/
theorem run_overshoot_bound (ceiling cmax : ℕ) :
    ∀ (spent : ℕ) (costs : List ℕ), spent ≤ ceiling + cmax → (∀ c ∈ costs, c ≤ cmax) →
      runSpend ceiling spent costs ≤ ceiling + cmax
  | spent, [], h, _ => by simpa [runSpend] using h
  | spent, c :: cs, h, hc => by
    rw [runSpend]
    split
    · rename_i hlt
      have hcm : c ≤ cmax := hc c (List.mem_cons_self ..)
      exact run_overshoot_bound ceiling cmax (spent + c) cs (by omega)
        (fun x hx => hc x (List.mem_cons_of_mem _ hx))
    · exact h

/-- The bound as an operator states it: a run that starts at zero spend, under a ceiling of
    `limit + granted`, ends at most one worst-case call past that ceiling. -/
theorem run_bounded_by_limit_grants_and_one_call (limit granted cmax : ℕ) (costs : List ℕ)
    (hc : ∀ c ∈ costs, c ≤ cmax) :
    runSpend (limit + granted) 0 costs ≤ limit + granted + cmax :=
  run_overshoot_bound (limit + granted) cmax 0 costs (Nat.zero_le _) hc

/-- **The bound is tight** — the overshoot really can reach a full `cmax`, so the `+ cmax` is not
    slack that a sharper proof could remove. One call of cost `cmax` admitted at `ceiling - 1`. -/
theorem run_overshoot_is_tight : runSpend 3 2 [5] = 7 := by decide

/-! ## Teeth (rung-1): a buggy variant that drifts, caught by `decide` -/

/-- A grants-first divergence, modeled: consult grants FIRST and drop the fail-first check;
    a positive grant is consumed regardless of `on_exhaust`, so a fail-mode run that should have
    refused at the ceiling instead spends the grant. NOTE: `fail` is threaded but deliberately
    **never consulted** by the `match` below — that omission IS the bug being modeled. -/
def enforceBad (limit meter : ℕ) (fail : Bool) : ℕ → ℕ → List Grant → Outcome
  | granted, trips, gs =>
    if meter < limit + granted then Outcome.cleared granted trips
    else
      match gs with
      | [] => Outcome.parked trips
      | Grant.stop :: _ => Outcome.refused meter (limit + granted)
      | Grant.add 0 :: _ => Outcome.refused meter (limit + granted)
      | Grant.add (n + 1) :: rest =>
        enforceBad limit meter fail (granted + (n + 1)) (trips + 1) rest

def enforceBadMeasured (limit meter : ℕ) (fail : Bool) (gs : List Grant) : Outcome :=
  enforceBad limit meter fail 0 0 gs

/-- **The buggy variant diverges** — `decide`-checked, axiom-free. With `fail=true`, over the ceiling,
    and a positive grant delivered, the correct transition refuses at the ceiling (fail-first) while
    the buggy one would consume the grant. The witness makes the two return different outcomes. -/
theorem enforceBad_diverges :
    ∃ (limit meter : ℕ) (gs : List Grant),
      enforceMeasured limit meter true gs ≠ enforceBadMeasured limit meter true gs := by
  refine ⟨3, 3, [Grant.add 5], ?_⟩
  decide

/-! ## Conformance vectors — the machine-derived rows every interpreter is held to

Each row is `(limit, meter, fail, grants) ↦ outcome`, verified against the model by `decide`
(so the table cannot drift from the transition it documents). `Effective/enforce_vectors.json` is
the same table emitted for the Python conformance test (`tests/test_enforce_measured_conformance.py`),
which runs the LIVE `enforce_measured` against these rows — the reference made executable (A4/A5). -/

/-- Inputs to one conformance row. -/
structure Vec where
  limit : ℕ
  meter : ℕ
  fail : Bool
  grants : List Grant
  deriving Repr

/-- The canonical rows: one per decision-table arm, plus the adversarial/boundary cases. -/
def conformanceVectors : List (Vec × Outcome) :=
  [ -- under the ceiling: cleared with no trip
    (⟨3, 2, false, []⟩, Outcome.cleared 0 0),
    (⟨3, 2, true, []⟩, Outcome.cleared 0 0),
    -- at the ceiling, park-mode, no grant: the base park (trip 0)
    (⟨3, 3, false, []⟩, Outcome.parked 0),
    -- over the ceiling, park-mode, no grant: still parks at trip 0
    (⟨3, 5, false, []⟩, Outcome.parked 0),
    -- fail-first: refuses at the ceiling before consulting grants
    (⟨3, 3, true, [Grant.add 5]⟩, Outcome.refused 3 3),
    (⟨3, 5, true, []⟩, Outcome.refused 5 3),
    -- stop / zero grants refuse
    (⟨3, 3, false, [Grant.stop]⟩, Outcome.refused 3 3),
    (⟨3, 3, false, [Grant.add 0]⟩, Outcome.refused 3 3),
    -- a positive grant refills and clears (trip advanced to 1)
    (⟨3, 3, false, [Grant.add 5]⟩, Outcome.cleared 5 1),
    -- a positive grant that is insufficient re-parks at the NEXT trip (1)
    (⟨3, 5, false, [Grant.add 1]⟩, Outcome.parked 1),
    -- two grants: first refills but still over, second clears (trip advanced to 2)
    (⟨3, 5, false, [Grant.add 1, Grant.add 2]⟩, Outcome.cleared 3 2),
    -- a generous grant clears in one refill (trip 1)
    (⟨3, 5, false, [Grant.add 3]⟩, Outcome.cleared 3 1) ]

/-- **Every conformance vector holds against the model** — `decide`-checked, axiom-free, so the
    exported table is a machine-verified projection of `enforceMeasured`. -/
theorem conformance_vectors_hold :
    ∀ vo ∈ conformanceVectors,
      enforceMeasured vo.1.limit vo.1.meter vo.1.fail vo.1.grants = vo.2 := by
  decide

/-! ## Trust check -/

#check @enforceMeasured
#print axioms under_ceiling_clears
#print axioms fail_refuses_before_grants
#print axioms park_when_no_grant
#print axioms stop_refuses
#print axioms zero_grant_refuses
#print axioms positive_grant_advances
#print axioms run_overshoot_bound
#print axioms run_bounded_by_limit_grants_and_one_call
#print axioms run_overshoot_is_tight
#print axioms enforceBad_diverges
#print axioms conformance_vectors_hold

end Effective.EnforceMeasured
