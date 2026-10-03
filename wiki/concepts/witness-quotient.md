# The witness is a quotient of the store

A race branch records a digest of the values it was handed, so a retry that reproduces its ending
can be told from one that does not. That digest is only as good as the equality behind it, and the
equality is not a matter of taste: it is fixed by what durability keeps.

## The three maps

Write a leaf op's journey as three functions of the value the domain returned.

| map | what it is | where |
|---|---|---|
| `encode` | `to_jsonable_python(value)`, the checkpoint form | `handlers/absurd._dump` |
| `keep` | what the engine holds of that: `jsonb` on Absurd, JSON text on SQLite | the engine |
| `load` | the op's declared type put back on, `TypeAdapter(schema).validate_python` | `handlers/absurd._load` |

A branch is handed `load(encode(v))` on the attempt that ran the op, since a step returns through
`load` whether it ran the thunk or served a checkpoint. It is handed `load(keep(encode(v)))` on the
attempt that replays it. **The live value never reaches the workflow**, and therefore never reaches
the witness: the only difference between what the two attempts see is what `keep` did in between.

## What the codec owes

Let `W` be `witness`, and let `E = keep ∘ encode` be what the store holds.

**Stability.** `W(load(encode(v))) = W(load(E(v)))`, for every value a checkpoint can hold.

That is the whole runtime obligation, and it implies the half of fidelity that matters. If two
values are stored alike then they are loaded alike, so they witness alike: **`ker(E) ⊆ ker(W)`**,
the witness is never finer than the store. A witness that were finer would fail a retry that was
handed the same input, which is the defect this arc started from, on Postgres, for any branch
handed a dict.

So `W` factors through `E`: there is a `w` with `W = w ∘ load ∘ decode ∘ E`. The codec is a
function of what durability records: the rule that a branch's identity is what the store keeps,
stated as an equation rather than as a sentence.

**Coarsening.** Where `w` is not injective, two values the store keeps apart witness alike, and a
retry handed the second could complete carrying the first. The ruling permits this and does not
leave it implicit: every class is declared, and a class nobody declared fails the gate.

| class | merged | kept apart by |
|---|---|---|
| a number's width | `1`, `1.0`, an `IntEnum` member, an `int` subclass | both engines |
| a zero's sign | `0`, `0.0`, `-0.0` | SQLite; Absurd drops the sign itself |
| a mapping's writing order | the same members written in two orders | SQLite; Absurd sorts keys itself |

Each row is a place the shared quotient is coarser than the finer engine, which is the price of
one digest serving two stores. A number used as a mapping key escapes the first two rows, since the
encoding turns the key into its text and `"1"` and `"1.0"` are different keys.

## What has no witness at all

Three outcomes sit outside the quotient, and only the third is about the codec.

| outcome | what happens |
|---|---|
| the encoding refuses the value (bytes no text covers) | no op could record it; the step fails before a digest matters |
| the store refuses the encoding (a NUL in a string, on `jsonb`) | the same failure one layer down |
| the codec takes no witness (a value holding itself) | the digest is `None`, so the retry fails loudly rather than trusting a partial one |

## What the sweep covers, and what it does not

`scripts/witness_sweep.py` builds a product of leaves and positions and measures every cell against
both engines, so a shape nobody recalled is in the space by construction. Its declared classes are
pinned as properties W1 to W3, and a shape an earlier round found by hand
is pinned present, since a sweep licenses a claim only over what it covered
([[concepts/enforcer-domain]]).

Two things it does not reach. It measures one process, so a witness that followed this process's
hash seed is a separate subject, not the sweep's. And it measures a leaf op's value, where a branch is
also handed refusals, a race's own choice and its children's digests, which take the same codec by
other call paths.

**It is an instrument, so it was itself wrong first** ([[concepts/evidence]]). Its first version
witnessed the live domain value and reported ninety-three failures against a codec that was
correct, because a handler hands a branch an encoded value on both attempts. A pinned test had the
same defect: a measurement of a call path nothing takes. The rule it teaches is to derive what the
code hands the function under test, rather than what the prose around it says the function is for.
