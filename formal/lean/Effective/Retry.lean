import Mathlib.Data.List.Basic

/-!
# Effective: retries, the tape as an effect, and the edge's decision

A task fails at once on an error a retry would raise again. Two facts carry that rule.

* **A run that read only the record re-runs identically.** The tape is an effect: a read answers
  `hit v` when the record holds `v`, and otherwise whatever the environment answers, a miss whose
  thunk runs, a fault, or a value another writer put there. User code is a function of the values
  it is handed. So a run every one of whose reads hit is a function of the record alone, and a
  retry against the same record, under any environment, takes the same reads to the same end
  (`replay_without_fresh_or_unrecorded_reraises`). This generalizes `Step.replay_stable`, whose
  lookup is the `hit` case.
* **The edge's decision is total** (`failingLeaf`), over what the handler knows of a raised error
  and the attempt that raised it. Its rows are emitted as `retry_vectors.json` and the Python test
  runs each through the live `failing_leaf`.

Each read the handler makes is a row of `src/effective/handlers/transitions.py`, labeled `record`,
`fresh` or `world`. `edgeSilent` says which labels leave a run's counts at zero: `record`, and a
`world` read kept by suspending, by a monotone clock or by eliding a read. That such a read answers
as the record would is each row's argument in the table, assumed here, which is where the first
fact meets the labels.
-/

namespace Effective.Retry

abbrev Name := Nat
abbrev Val := Nat

/-- What a read of the tape answers. -/
inductive TapeResult
  | hit (v : Val)
  | miss
  | fault
  | foreign (v : Val)
  deriving DecidableEq, Repr

/-- What the environment answers a read the record does not hold: a miss whose thunk runs, a
    fault, or a value another writer put there. A hit is the record's alone. -/
inductive Answer
  | miss
  | fault
  | foreign (v : Val)
  deriving DecidableEq, Repr

def Answer.result : Answer → TapeResult
  | .miss => .miss
  | .fault => .fault
  | .foreign v => .foreign v

/-- A read: the record's value when it holds one, the environment's answer otherwise. -/
def read (tape : Name → Option Val) (env : Name → Answer) (n : Name) : TapeResult :=
  match tape n with
  | some v => .hit v
  | none => (env n).result

/-- **A record read is a function of the record, given a hit.** -/
theorem read_hit_is_the_record (tape : Name → Option Val) (env env' : Name → Answer)
    (n : Name) (v : Val) (h : tape n = some v) : read tape env n = read tape env' n := by
  simp [read, h]

/-- A deterministic program: the next read it makes, or `none` when it ends, and whether it ends
    by raising, each a function of the values it has been handed. -/
structure Program where
  next : List Val → Option Name
  raises : List Val → Bool

/-- How a run ends: it finished, raising or returning, on the values it was handed; it met an
    answer other than the record at a read; or it ran out of fuel. -/
inductive End
  | finished (raised : Bool) (vals : List Val)
  | departed (n : Name) (r : TapeResult) (vals : List Val)
  | exhausted
  deriving DecidableEq, Repr

/-- One run: read until the program ends, stopping at the first read that is not a hit. -/
def run (p : Program) (tape : Name → Option Val) (env : Name → Answer) :
    Nat → List Val → End
  | 0, _ => .exhausted
  | fuel + 1, vals =>
    match p.next vals with
    | none => .finished (p.raises vals) vals
    | some n =>
      match read tape env n with
      | .hit v => run p tape env fuel (vals ++ [v])
      | r => .departed n r vals

/-- **A run that read only hits, and ended, re-runs identically under any environment.** The
    retry rule's first fact: an attempt whose frame ran nothing fresh and met no answer other
    than the record ends where a retry against the same record ends. -/
theorem replay_without_fresh_or_unrecorded_reraises (p : Program) (tape : Name → Option Val)
    (env env' : Name → Answer) (fuel : Nat) (vals : List Val) (raised : Bool)
    (out : List Val) (h : run p tape env fuel vals = .finished raised out) :
    run p tape env' fuel vals = .finished raised out := by
  induction fuel generalizing vals with
  | zero => simp [run] at h
  | succ k ih =>
    simp only [run] at h ⊢
    cases hn : p.next vals with
    | none => simpa [hn] using h
    | some n =>
      rw [hn] at h
      cases ht : tape n with
      | none =>
        simp only [read, ht] at h
        cases henv : env n <;> simp_all [Answer.result]
      | some v =>
        simp only [read, ht] at h ⊢
        exact ih (vals ++ [v]) h

/-! ## The labels, and what leaves a run's counts at zero -/

/-- Why a `world` read stays sound, as the transition table states it. -/
inductive Kept
  | counted
  | settled
  | suspends
  | monotone
  | elides
  deriving DecidableEq, Repr

inductive Label
  | record
  | fresh
  | world (k : Kept)
  deriving DecidableEq, Repr

/-- A label whose transition leaves both of a run's counts at zero: neither a thunk it ran nor
    a substrate read it counted. -/
def edgeSilent : Label → Bool
  | .record => true
  | .fresh => false
  | .world .counted => false
  | .world .settled => false
  | .world _ => true

/-- A run whose every transition is silent ran nothing fresh and counted nothing. -/
theorem silent_labels_are_record_or_uncounted_world (l : Label) (h : edgeSilent l = true) :
    l = .record ∨ l = .world .suspends ∨ l = .world .monotone ∨ l = .world .elides := by
  cases l with
  | record => simp
  | fresh => simp [edgeSilent] at h
  | world k => cases k <;> simp_all [edgeSilent]

/-! ## The edge's decision -/

/-- What the handler knows of one leaf of a raised error. -/
structure Leaf where
  typed : Bool        -- an `Unretryable`
  rederived : Bool    -- raised by a frame that ran nothing fresh and counted no substrate read
  refusal : Bool
  undecided : Bool    -- inside a race no branch is left to win
  deriving DecidableEq, Repr

/-- The attempt that raised it. -/
structure Attempt where
  delayed : Bool      -- a retry after it waits before it runs
  final : Bool        -- no retry follows it
  deriving DecidableEq, Repr

/-- A leaf a retry would raise again. -/
def futile (a : Attempt) (l : Leaf) : Bool :=
  (l.typed && !l.undecided) || (l.rederived && !a.delayed)

/-- The index of the leaf the task fails of on this attempt, or `none` while a retry could still
    succeed. Mirrors `handlers.base.failing_leaf`, over the leaves in order. -/
def failingLeaf (a : Attempt) (ls : List Leaf) : Option Nat :=
  match ls.findIdx? (futile a) with
  | some i => some i
  | none => if (ls ≠ [] && ls.all (·.refusal)) || (a.final && ls ≠ []) then some 0 else none

theorem futile_leaf_fails_now (a : Attempt) (ls : List Leaf) (i : Nat)
    (h : ls.findIdx? (futile a) = some i) : failingLeaf a ls = some i := by
  simp [failingLeaf, h]

/-- **A delayed retry keeps a rederived error's attempts**: before the last attempt, only a typed
    leaf outside an undecided race, or a failure of refusals alone, fails a task whose retry
    waits. -/
theorem delayed_fails_only_typed (ls : List Leaf)
    (hnone : ∀ l ∈ ls, l.typed = false ∨ l.undecided = true)
    (hrefusal : ∃ l ∈ ls, l.refusal = false) :
    failingLeaf ⟨true, false⟩ ls = none := by
  have hf : ls.findIdx? (futile ⟨true, false⟩) = none := by
    rw [List.findIdx?_eq_none_iff]
    intro l hl
    rcases hnone l hl with h | h <;> simp [futile, h]
  obtain ⟨l, hl, hr⟩ := hrefusal
  have hall : ls.all (·.refusal) = false := by
    rw [List.all_eq_false]; exact ⟨l, hl, by simp [hr]⟩
  rw [failingLeaf, hf]
  simp [hall]

/-- **A typed leaf inside an undecided race does not fail the task before its last attempt**: a
    retry may run a branch fresh and win. -/
theorem undecided_typed_waits (a : Attempt) (l : Leaf) (hu : l.undecided = true)
    (hr : l.rederived = false) (hf : l.refusal = false) (hfinal : a.final = false) :
    failingLeaf a [l] = none := by
  simp [failingLeaf, futile, hu, hr, hf, hfinal]

/-- **A failure of refusals alone fails at once**, whether or not the retry waits. -/
theorem refusals_fail_now (a : Attempt) (ls : List Leaf) (hne : ls ≠ [])
    (hall : ∀ l ∈ ls, l.refusal = true) : (failingLeaf a ls).isSome = true := by
  unfold failingLeaf
  split
  · simp
  · have : ls.all (·.refusal) = true := List.all_eq_true.mpr (fun l hl => by simp [hall l hl])
    simp [this, hne]

/-- **The last attempt fails of its first leaf** when nothing earlier decided. -/
theorem final_fails_first (ls : List Leaf) (delayed : Bool) (hne : ls ≠ [])
    (hnone : ls.findIdx? (futile ⟨delayed, true⟩) = none) :
    failingLeaf ⟨delayed, true⟩ ls = some 0 := by
  simp [failingLeaf, hnone, hne]

/-! ## Conformance vectors -/

def bools : List Bool := [false, true]

def leaves : List Leaf :=
  bools.flatMap fun t => bools.flatMap fun r => bools.flatMap fun f => bools.map fun u =>
    ⟨t, r, f, u⟩

def attempts : List Attempt := bools.flatMap fun d => bools.map fun f => ⟨d, f⟩

/-- Every attempt against every failure of one or two leaves, with the leaf it fails of. The
    rows are `failingLeaf`'s own answers, so they carry no claim of their own: they hold the live
    Python to this function, whose properties are the theorems above. -/
def conformanceVectors : List (Attempt × List Leaf × Option Nat) :=
  attempts.flatMap fun a =>
    (leaves.map (fun l => [l]) ++ leaves.flatMap fun l => leaves.map fun m => [l, m]).map
      fun ls => (a, ls, failingLeaf a ls)

/-! ## Trust check -/

#print axioms replay_without_fresh_or_unrecorded_reraises
#print axioms read_hit_is_the_record
#print axioms silent_labels_are_record_or_uncounted_world
#print axioms delayed_fails_only_typed
#print axioms final_fails_first
#print axioms futile_leaf_fails_now
#print axioms undecided_typed_waits
#print axioms refusals_fail_now

end Effective.Retry
