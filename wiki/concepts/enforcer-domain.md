# A gate is bounded by what it scans

The sentence describing a gate tends to claim more than the gate covers. A denylist is bounded by
what it enumerates, and a gate by what it scans, so before trusting a green, ask what the enforcer
reaches. The remedy is the same for both: key on the **resolved** thing, never on the text that
names it.

## Where the domain falls short of the claim

| where | the question | a case |
|---|---|---|
| the grammar | what does the enforcer's pattern match? | a lint for a key rebuilt by joining rendered terms that matches one spelling of the join misses the literal `";".join(...)` its own docstring gives as the defect |
| the resolver | what would a different machine answer? | a citation check that asks the filesystem resolves a gitignored file on the machine that built it and in no clone. Run the gate in a fresh clone and compare the verdicts |
| the renderer | what did the view collapse? | a view that prints one representative site per key template hides every template minted at more than one place, which is the duplication it exists to show |
| the citation | what does it resolve? | a check of `file.py:1513` resolves the path and never the line, so it grades the same as `file.py:1` while the cited code moves |

Naming the world a resolver consults makes a second surface, and it needs its own check. A declared
exemption for a generated artifact is a hole in the domain, so the declaration is checked too: the
file is still untracked, something still cites it, and its producer still exists.

## A domain written as spellings

A domain written as the spellings of one referent is always one spelling short, because the
referent grows new spellings and nothing links them. When a domain keeps coming up one entry
short, ask whether the property it tests is closed under its own application. A module that
yields through an effect module is itself an effect module, so the lint's set of effect modules is
a fixpoint over the scanned files (`lint._effect_module_closure`), where a list would always lag.

## A zero is a claim about the enumerator

A report of absence first claims that the enumerator could see the construct. A sweep that cannot
see a keyword class pattern reports zero uses of everything written that way, and a "zero uses,
delete it" pass on that sweep proposes deleting exactly that code. So
[`scripts/whouses.py`](../../scripts/whouses.py) enumerates with ast-grep rules, and reports an
inference miss as **UNRESOLVED**, a third outcome kept apart from "not a use". A tool's stated
ceiling is a claim too: run the query before believing it cannot be written.

## What a green licenses

- **A measurement licenses a claim over the domain it covered**, and no further.
- **An instrument is a claim**, and the one built most recently is the one least checked. A
  green-and-wrong instrument shows when you move what it describes; rerunning it shows nothing.
- **An engine more precise than the reference can launder a property into a green test.** Ask what
  a passing test relies on. Effective puts such a property in the durable name, as the generation
  coordinate in a `govern:` key, so both engines carry it.
- **A check that cannot fail is not a contract.** When no input this release can construct breaks
  a check, delete the check, and keep a test that fails the day a second case arrives.
- **A stub is not the system.** A `TypedDict` or a `.pyi` is a claim about a value, and coverage
  can be structurally unable to see a defect in code that never ran.
