import Effective.Decide

/-!
Emit `Effective.Permission.conformanceVectors` as JSON — the machine-derived conformance matrix
for the permission cascade's decision. Every row is `decide`-verified against the Lean model
(`conformance_vectors_hold`), so this output is a projection of the transition, not a hand table.

    lake exe decide_vectors > ../decide_vectors.json   (see `just formal-vectors`)

The Python conformance test (`tests/test_decide_conformance.py`) runs the LIVE `permission.decide`
and the live `cascade` driver against these rows: the reference made executable.
The sibling of `Main.lean`, which does the same for the measured trip.
-/

open Effective.Permission

/-- A double-quote, kept out of the interpolation grammar. -/
def q : String := "\""

/-- A JSON `"key": value` pair (value pre-rendered). -/
def kv (k v : String) : String := q ++ k ++ q ++ ": " ++ v

def obj (fields : List String) : String := "{" ++ String.intercalate ", " fields ++ "}"

def str (s : String) : String := q ++ s ++ q

def verdictJson : Verdict → String
  | .allow => obj [kv "kind" (str "allow")]
  | .deny r => obj [kv "kind" (str "deny"), kv "reason" (toString r)]
  | .escalate => obj [kv "kind" (str "escalate")]

def verdictsJson (vs : List Verdict) : String :=
  "[" ++ String.intercalate ", " (vs.map verdictJson) ++ "]"

def decisionJson : Decision → String
  | .allow => obj [kv "kind" (str "allow")]
  | .deny r => obj [kv "kind" (str "deny"), kv "reason" (toString r)]

def rowJson (vd : Vec × Decision) : String :=
  "  " ++ obj [kv "verdicts" (verdictsJson vd.1.verdicts),
    kv "default" (verdictJson vd.1.fallback), kv "decision" (decisionJson vd.2)]

def main : IO Unit :=
  IO.println ("[\n" ++ String.intercalate ",\n" (conformanceVectors.map rowJson) ++ "\n]")
