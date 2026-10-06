# ADR-0023: Test roles: what a test is for decides what its pass proves

- **Date:** 2026-08-04
- **Status:** Accepted. `pyproject.toml` registers five role markers under `--strict-markers`, a
  test with no role is a `unit` test, and every other role is declared. The set is open: a new
  role is expected, with BDD a named candidate.
- **Relates to:** ADR-0016 (the three-tier proof partition; this names the test tier's internal
  structure and adds a second Lean-to-tests edge), ADR-0009 (the cross-backend conformance suite,
  the `conformance` role's exemplar).
- **Context:** an adversarial test caught a defect in an approval gate that every confirmatory test
  had passed. It could not have been written as coverage, and the rules the repo applied to tests
  had no place for it.

## 1. The decision

**A test has a `role`, the role decides what a passing run proves, and a rule that applies to one
role need not apply to another.**

`role` is the repo's word for this move one population over: `effective.lint` keeps
`WORKFLOW_ROLE_SRCS`, `is_workflow_role` and `--role-coverage`, whose `check_role_coverage` is a
completeness check over a role partition (every file that yields an effect op is workflow-role,
exempt, or a violation). A test's role deciding which rules apply is the same idea.

## 2. Why a vocabulary is needed

Two rules the repo follows are wrong when applied globally, and a role vocabulary is what makes
each statable correctly.

### 2a. "Minimize overlap" is a unit-role rule

It is right for unit tests and wrong for `tests/test_composition.py` and `tests/_conformance.py`,
whose overlap is the product: one law over every ordered pair of combinator holes, one workflow over two
engines. An overlap query across all roles flags the strongest suites in the repo as redundant.

### 2b. A pass is evidence in proportion to what could have failed

| role | what a pass proves |
|---|---|
| journey | the path works, on its own |
| adversarial | the attack failed, which is evidence only if the test could have failed |

So **an adversarial test carries a mutation obligation**: break the implementation and confirm the
test reddens. A journey does not owe one.

The motivating case: the `human` permission tier named its park after the gated op alone, so two
occurrences of one tool in a run parked on one name and a single approval settled both. Every
"does the tier park?" test passed. Only "can approving \$5 authorize \$5,000,000?" failed. The
name now carries a per-run occurrence suffix, and
`test_one_approval_does_not_authorize_a_second_charge_of_the_same_tool` (`tests/test_permission.py`)
pins it.

## 3. The five roles

| role | overlap with other roles | what a pass proves | obligation |
|---|---|---|---|
| **unit** | waste: minimize it | this seam behaves | none |
| **spine** | the deliverable | the pieces compose | none |
| **journey** | deliberate | a realistic path works end to end | none |
| **adversarial** | irrelevant | the attack failed, if the test could have failed | a mutation check |
| **conformance** | by design, across arms | the implementations agree with the model | enroll every interpreter |

The discriminator is what overlap means and what a pass proves. Subject matter does not decide it:
two roles may exercise the same code.

`journey` and `adversarial` share machinery, since both drive realistic multi-step scenarios, and
their acceptance conditions are opposite. A journey confirms; an adversarial test falsifies. A
journey that cannot fail is a broken journey; an adversarial test that cannot fail is
indistinguishable from a passing one.

## 4. The exemplars

| file | role |
|---|---|
| `tests/test_decide_conformance.py`, `tests/test_govern_conformance.py`, `tests/test_enforce_measured_conformance.py` | conformance: a Lean model's rows over every interpreter (§5) |
| `tests/test_conformance.py` with `tests/_conformance.py` | conformance: one workflow set, two engines (ADR-0009) |
| `tests/test_composition.py` with `tests/_composition.py` | conformance: a table over every ordered pair of combinator holes |
| `tests/test_funnel_triage_example.py` | journey: the workflow in `tests/_funnel.py` runs five combinators along two arms, the deepest composition the suite drives |
| `tests/test_cart_checkout_example.py` | journey, over the canonical record: a ledger row per line item at one program point |
| `tests/test_mcts_search_example.py` | journey, over a workflow whose control flow is decided by accumulated recorded values |
| `tests/test_coding_machine_example.py` | journey, over a workflow with real backedges, so its projection is a state diagram |
| `tests/test_projection.py` | spine: the projections against the workflows above, since a quotient means something only over tapes a workflow produced |
| `tests/test_op_key_injectivity.py` | adversarial: injectivity attacks |
| `tests/test_permission.py` (the approval attack) | adversarial |

Each declares its role with a marker. A workflow a test drives lives under `tests/`, as
`tests/_funnel.py` does, so the determinism-boundary lint (whose `WORKFLOW_ROLE_SRCS` is
path-keyed) and `ty` both reach it.

### 4a. Sampled properties

A sampled property runs one workflow or combinator over generated cases and judges each by
invariants, where a journey asserts one realistic path. `tests/test_funnel_sweep.py` (over the
funnel) and `tests/test_descend_sweep.py` (over `descend`'s budget, grant and depth space) are the
exemplars. Two rules keep a sweep from being a slower journey test:

| rule | why |
|---|---|
| the oracle is written before the generator, and passes on the cases whose answers are known | invariants written after the cases are a mirror of whatever the generator produced |
| each invariant names the mutation that reddens it | an invariant nothing can falsify reads as coverage and provides none |

A sweep earns its place when the sampled space reaches shapes the scripted case cannot, and it
states that executably: each file's `test_the_seeds_reach_the_corners_…` names every corner, so a
generator change that stops producing one fails there.

A green sweep is evidence about its seeds until something shows they discriminate, and the mutation
round is that something. Run the mutations against the generated cases as well as the scripted
ones: on `tests/test_descend_sweep.py` the same mutations redden several times as many generated
cases as scripted ones, and that ratio is what says the sampling does work.

`tests/test_key_sweep.py` applies the same discipline to a reporting tool: `scripts/key_sweep.py`
had two rules that were wrong in ways no run of it could show, so each case there records the
mutation that reddens it.

## 5. The second Lean-to-tests edge: the model as a row generator

ADR-0016 partitions assurance into three tiers, tests the third. Where a proof tier stops, it hands
the next tier a named assumption (`(A-serialize)`, `(A-canon)` in `formal/lean/Effective/Keys.lean`),
which a parameter grid discharges by sampling. A second, stronger edge is the `conformance` role's
definition:

1. a `decide`-checked table lives in the `.lean` file (`conformance_vectors_hold`, axiom-free), so
   the table cannot drift from the transition it documents;
2. a `lean_exe` renders it as JSON;
3. `just formal-vectors` writes `formal/{enforce,decide,govern}_vectors.json`, which are committed;
4. Python parametrizes every live interpreter over the committed rows.

ADR-0016's edge is a named assumption plus adversarial sampling; this one is machine-derived rows.
Both are live.

Two rules travel with it:

| rule | why |
|---|---|
| enroll every interpreter | the drift lives on the one not enrolled |
| name the model-to-code type gap as a sampling discharge, tagged `(A-<name>)` in the `.lean` header and the test docstring | `(A-money)`, `(A-reason)` and `(A-tags)` are the cases |

**Open:** nothing checks a committed vector file against a fresh regeneration. The `decide` theorem
keeps the table faithful to the model; a stale `.json` against an edited `.lean` is caught only by
re-running `just formal-vectors`.

## 6. Mechanics

Markers are registered in `pyproject.toml`, and `--strict-markers` refuses an unregistered one.
A test that declares no role is a `unit` test: a collection hook in `tests/conftest.py` marks it,
so `-m unit` selects every test whose pass proves one seam in isolation, and a test that proves
more declares `spine`, `journey`, `adversarial` or `conformance`. The default is the common case,
and the declaration marks the exception.

`just cov-contexts` (`scripts/cov_contexts.py`, under `COVERAGE_CORE=pytrace`) records which test
ran each line, so the unit-role overlap rule is queryable. Scope the query by marker, or it indicts
the conformance suites it should exempt.

Overlap is a proxy. A coverage context says a line ran under a test; it cannot say the assertion
was about that line. Two unit tests on one line may test different behaviors, and a test can cover
a branch it does not exercise (a `zip` that ran out before the index the assertion meant to reach
let a mutation survive a pin). The overlap query finds review candidates, never defects.

## 7. The same move, one population over: primitive, convenience, minter

Overlap against a declared role is the defect; overlap alone is not. The substrate has three roles
of its own:

| role | what it is | overlap with | normal form |
|---|---|---|---|
| **primitive** | a `WorkflowOp` member: the vocabulary every handler is total over | another primitive: waste | the union itself |
| **convenience** | delegates to a primitive, adding no interpretive power | a primitive: the deliverable; another convenience: waste | a body that is one `yield from` of a primitive |
| **minter** | a pure function whose output is an identity | two minters that agree on every input are one minter | the key template, in `build/key-registry.json` |

`govern` and `serve` are the op-seam primitives (ADR-0019); `ask_llm` and `call_tool` are
conveniences over `step`. A minter exists so a namespace is composed in one place a reader can
grep, and so a test asks for the name rather than spelling its bytes. A test parameterized on a
different minter from production measures the copy; `react.tool_key` delegates to
`api.direct_tool_key` for that reason.

Declare the exemplars and gate nothing, for §6's reason: a proposed classification of minters
measured about three in four correct, and a proposal set that wrong must not gate.

## 8. Consequences

- The roles are an open set. Adding one is expected and needs no schema change.
- The mutation craft is general; this ADR assigns the obligation to a role.
- The unit-role overlap rule is statable, scoped by marker.
- ADR-0016 stands: this adds a second edge to its Lean-to-tests boundary beside the first.
