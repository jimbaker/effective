# Keys are partially ordered by coverage

The key grammar was built for injectivity: two distinct names never render to one string, and a
string parses to one structure. Keys also turn out to be partially ordered, by `Key.covers`, and
that order is what an allow-set is read in. The order follows from the grammar, and
`PermitPolicy` reads it where a text match would alias.

## The order

`a.covers(b)` holds when `b` is `a` or a narrowing of it:

| a | b | covers | why |
|---|---|---|---|
| `tool` | `tool:read_file,0` | yes | a bare tag covers every key under it |
| `tool:a` | `tool:a,1` | yes | a leading run of the last term's coordinates |
| `step;tool:a` | `step;tool:a,1` | yes | the leading terms are equal |
| `tool:a` | `tool:a#2` | yes | an unnumbered key covers every occurrence |
| `tool:a#2` | `tool:a#2` | yes | a numbered key covers itself alone |
| `tool:a#2` | `tool:a` | no | |
| `tool:a` | `tool:ab` | no | a different atom, which a text prefix would have matched |

It is reflexive and transitive by construction, and antisymmetric because keys compare by
structure: if each covers the other, they have the same terms and the same occurrence, so they are
equal. **Injectivity is what makes it an order on names**: one structure renders to one stored
text and back, so ordering structures orders the names the engines store, and sorting entries by
their stored text breaks no tie between two distinct entries.

| coordinates | how `covers` reads them |
|---|---|
| bare and named alike | positionally, as written: `tool:a` covers `tool:a,n=x` |
| a named one left out in the middle | a different position, never "any": `govern:g,r1` covers `govern:g,r1,generation=1,pass-n=1`, and `govern:g,r1,pass-n=1` does not |
| the same names in another order | different keys: `tool:a,b=1` does not cover `tool:a,c=2,b=1` |

The occurrence rule reads the substrate's two scopes as an order. A `SETTLEMENT` namespace numbers
a second ask (`govern:r1#2`) because one approval settles one op, so a numbered key names one
question and covers nothing else. An unnumbered key names the identity every ask of it shares, the
way an `ACCRUAL` grant is read. Ignoring the occurrence would have made `tool:a` and `tool:a#2`
cover each other, which breaks antisymmetry. That half of the order applies to awaits: a tool
call's placed key carries its position in coordinates and never a `#N`, and `#1` has no written
form, so an entry can name the second ask of an await alone and never the first alone.

## What the order buys an allow-set

A `PermitPolicy` permits every key some entry covers, and holds its entries in **canonical form**:
the entries no other entry covers, sorted by their stored text.

| law | follows from |
|---|---|
| the canonical form permits exactly what the written entries permit | a dropped entry is covered by a kept one, and coverage is transitive |
| any order, duplicate or redundant entry gives the same policy | the uncovered entries of a set do not depend on how it was listed |
| two allow-sets that permit the same keys are one policy | every entry is a key that covers itself; antisymmetry makes the uncovered entries unique |

So policy equality is coverage equality, which is what tuning compares: two candidates that permit
the same keys are one point on the frontier. The order is the decision; the sort only fixes the
stored bytes, since a `frozenset`'s JSON order follows per-process string hashing.

`tests/test_allow_set_laws.py` pins each law over every key a small grammar generates and every
allow-set of up to three of them, with all their permutations, and the partial order over a wider
set that adds named coordinates. It is a `property` suite, and each test was seen failing against
a mutant: an unsorted form, a form that keeps covered entries, a `covers` that ignores the
occurrence, one that compares only the tag, one blind to coordinate names, and a `permits` that
reads only the first entry.

## Lifting it to policies

The order extends to whole policies: `P ≤ Q` when every entry of `P` is covered by some entry of
`Q`. On canonical forms it is a partial order too, so `P ≤ Q` and `Q ≤ P` mean `P == Q`. That is
the shape of attenuation's law, a child's permissions within its parent's, which makes the key
algebra a candidate foundation for the authority algebra. No test pins `≤`, and no formal model
states it.

## A step's key names its kind

`step_key` puts a step's op kind inside the `step` arm, so the kinds are regions of the order:
`step;llm` covers every `AskLLM` step, `step;judge` every judgment, and neither covers a tool step.

| the step's name | its key |
|---|---|
| an atom: `ask_llm("extract")`, `call_tool("search")`, `judge("triage")` | `step;llm:extract`, `step;tool:search`, `step;judge:triage` |
| a minter's unnumbered `Key`: a ReAct turn, a spawn, an interrupt poll | `step;llm;react:turn`, `step;tool;spawn:child`, `step;tool;interrupt:pre` |
| text holding key structure, or a `Key` carrying an occurrence | refused (`FormattedStepName`, `NumberedStepName`) |

Context goes in a scope or a frame (`rec:3;step;llm:summarize`). A minter's tag is owned by one
package, and `--key-registry` refuses a registered tag minted in a second one.

**The limit.** An author's own composed `Key` passed as a step's name can borrow another minter's
tag within its kind: `compose_key(t"react:turn")` in author code records `step;llm;react:turn`,
a ReAct turn's key. The ownership check reads this tree's minters, so it proves Effective's own
code never borrows a tag, and an author can still write one.

## Related

[concepts/witness-quotient](witness-quotient.md) is the other place an equivalence over keys and
values is fixed by structure rather than taste. [concepts/flatten](flatten.md) is why a key has no
text form to match on.
