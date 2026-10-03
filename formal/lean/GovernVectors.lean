import Effective.Govern

/-!
Emit `Effective.Govern.conformanceVectors` as JSON — the machine-derived conformance matrix for
the `govern` gate's ruling. Every row is `decide`-verified against the Lean model
(`conformance_vectors_hold`), so this output is a projection of the transition, not a hand table.

    lake exe govern_vectors > ../govern_vectors.json   (see `just formal-vectors`)

The Python conformance test (`tests/test_govern_conformance.py`) runs the LIVE `govern.combine`
against these rows. The sibling of `Main.lean` (the measured trip) and `DecideVectors.lean` (the
permission cascade) — three transitions, one method.
-/

open Effective.Govern

def q : String := "\""

def kv (k v : String) : String := q ++ k ++ q ++ ": " ++ v

def obj (fields : List String) : String := "{" ++ String.intercalate ", " fields ++ "}"

def str (s : String) : String := q ++ s ++ q

def tags (ns : List Nat) : String :=
  "[" ++ String.intercalate ", " (ns.map toString) ++ "]"

def verdictJson : Verdict → String
  | .proceed => obj [kv "kind" (str "proceed")]
  | .park a => obj [kv "kind" (str "park"), kv "asks" (tags a)]
  | .refuse r => obj [kv "kind" (str "refuse"), kv "reasons" (tags r)]

def rowJson (vo : List Verdict × Verdict) : String :=
  "  " ++ obj [kv "verdicts" ("[" ++ String.intercalate ", " (vo.1.map verdictJson) ++ "]"),
    kv "ruling" (verdictJson vo.2)]

def main : IO Unit :=
  IO.println ("[\n" ++ String.intercalate ",\n" (conformanceVectors.map rowJson) ++ "\n]")
