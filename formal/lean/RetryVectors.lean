import Effective.Retry

/-!
Emit `Effective.Retry.conformanceVectors` as JSON: every attempt against every failure of one or
two leaves, with the leaf `failingLeaf` fails the task of.

    lake exe retry_vectors > ../retry_vectors.json   (see `just formal-vectors`)

`tests/test_retry_conformance.py` builds each failure as an exception tree and runs the live
`handlers.base.failing_leaf` against the row.
-/

open Effective.Retry

def q : String := "\""

def kv (k v : String) : String := q ++ k ++ q ++ ": " ++ v

def obj (fields : List String) : String := "{" ++ String.intercalate ", " fields ++ "}"

def flag (b : Bool) : String := if b then "true" else "false"

def leafJson (l : Leaf) : String :=
  obj [kv "typed" (flag l.typed), kv "rederived" (flag l.rederived),
    kv "refusal" (flag l.refusal), kv "undecided" (flag l.undecided)]

def rowJson (v : Attempt × List Leaf × Option Nat) : String :=
  "  " ++ obj [kv "delayed" (flag v.1.delayed), kv "final" (flag v.1.final),
    kv "leaves" ("[" ++ String.intercalate ", " (v.2.1.map leafJson) ++ "]"),
    kv "fails_of" (match v.2.2 with | some i => toString i | none => "null")]

def main : IO Unit :=
  IO.println ("[\n" ++ String.intercalate ",\n" (conformanceVectors.map rowJson) ++ "\n]")
