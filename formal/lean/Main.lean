import Effective.EnforceMeasured

/-!
Emit `Effective.EnforceMeasured.conformanceVectors` as JSON — the machine-derived conformance
matrix for the measured trip. Every row is `decide`-verified against the Lean model
(`conformance_vectors_hold`), so this output is a projection of the transition, not a hand table.

    lake exe enforce_vectors > ../enforce_vectors.json   (see `just formal-vectors`)

The Python conformance test (`tests/test_enforce_measured_conformance.py`) runs the LIVE
`enforce_measured` against these rows: the reference made executable.
-/

open Effective.EnforceMeasured

/-- A double-quote, kept out of the interpolation grammar. -/
def q : String := "\""

/-- A JSON `"key": value` pair (value pre-rendered). -/
def kv (k v : String) : String := q ++ k ++ q ++ ": " ++ v

def obj (fields : List String) : String := "{" ++ String.intercalate ", " fields ++ "}"

def grantJson : Grant → String
  | Grant.stop => obj [kv "stop" "true"]
  | Grant.add n => obj [kv "add" (toString n)]

def grantsJson (gs : List Grant) : String :=
  "[" ++ String.intercalate ", " (gs.map grantJson) ++ "]"

def outcomeJson : Outcome → String
  | Outcome.cleared g t => obj [kv "kind" (q ++ "cleared" ++ q), kv "granted" (toString g),
      kv "trips" (toString t)]
  | Outcome.parked tr => obj [kv "kind" (q ++ "parked" ++ q), kv "trip" (toString tr)]
  | Outcome.refused s c => obj [kv "kind" (q ++ "refused" ++ q), kv "spent" (toString s),
      kv "ceiling" (toString c)]

def rowJson (vo : Vec × Outcome) : String :=
  "  " ++ obj [kv "limit" (toString vo.1.limit), kv "meter" (toString vo.1.meter),
    kv "fail" (toString vo.1.fail), kv "grants" (grantsJson vo.1.grants),
    kv "outcome" (outcomeJson vo.2)]

def main : IO Unit :=
  IO.println ("[\n" ++ String.intercalate ",\n" (conformanceVectors.map rowJson) ++ "\n]")
