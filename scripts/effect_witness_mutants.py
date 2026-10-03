"""Run `tests/test_effect_witness.py` against named mutants of `scripts/effect_witness.py`.

The pins are an adversarial test role, so they owe a mutation check. Each mutant
reintroduces one defect the instrument once had or could have, by an exact source replacement; a
replacement that does not match exactly once is refused, so a mutant cannot pass by never being
applied. A mutant is written to `build/`, beside the repository root the script computes its paths
from, and the pins load it through `EFFECT_WITNESS_SOURCE`. The tracked script is never modified.

Exits non-zero if the unmutated pins fail, or if any mutant survives or never reaches a pin: only
a failing pin, pytest's exit code 1, counts as a kill.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "effect_witness.py"
MUTANT = ROOT / "build" / "effect_witness_mutant.py"
PINS = "tests/test_effect_witness.py"


@dataclass(frozen=True, slots=True)
class Mutant:
    name: str
    old: str
    new: str


MUTANTS = (
    Mutant("chain matched by type", "if value is op)", "if type(value) is type(op))"),
    Mutant("checkpoint hits written", "pending.written = bool(ran)", "pending.written = True"),
    Mutant(
        "re-appends written",
        "pending.written = ledger.conn.total_changes > before",
        "pending.written = True",
    ),
    Mutant(
        "caller's id over the stored id",
        "pending.identity = pending.identity or row.event_id.stored()",
        "pending.identity = row.event_id.stored()",
    ),
    Mutant(
        "artifact sensor gone",
        'effect = "artifact"\n                    identity = str(result)',
        'effect = "artifact"\n                    return result',
    ),
    Mutant(
        "drive_through left patched",
        "            setattr(owner, name, original)",
        "            if name != 'drive_through':\n                setattr(owner, name, original)",
    ),
    Mutant("injection never marked", "based.op is not drive.op", "False"),
    Mutant(
        "any op can write",
        '    """Whether executing `op` makes a write of this kind: total over `WorkflowOp`."""\n',
        '    """Whether executing `op` makes a write of this kind: total over `WorkflowOp`."""\n'
        "    return True\n",
    ),
    Mutant(
        "any store is owned",
        '    """Whether `store` is the one `handler` writes this kind of effect to."""\n',
        '    """Whether `store` is the one `handler` writes this kind of effect to."""\n'
        "    return True\n",
    ),
    Mutant(
        "a nested handler's drive ignored",
        "all(drive.op is based.drive.op for drive in later)",
        "True",
    ),
    Mutant(
        "a stacked witness's drive refused",
        "all(drive.op is based.drive.op for drive in later)",
        "not later",
    ),
    Mutant(
        "the unpatched drive_through wrapped",
        'driving(vars(module)["drive_through"])',
        'driving(__import__("effective.layers", fromlist=["x"]).drive_through)',
    ),
    Mutant("Join B drops side doors", 'row["site"] or row["writer"]', 'row["site"]'),
    Mutant(
        "the domain filter off",
        'return [row for row in rows if row["engine"] in IN_DOMAIN and row["written"]]',
        "return rows",
    ),
    Mutant("the cascade undeclared", '"effective.permission.cascade.<locals>.gate",', ""),
    Mutant(
        "convenience frames kept",
        'CONVENIENCE = frozenset({"src/effective/api.py", "src/effective/coroutine_api.py"})',
        "CONVENIENCE = frozenset()",
    ),
    Mutant(
        "delegates not skipped",
        'DELEGATES = frozenset({"step", "append", "_record_ledger"})',
        "DELEGATES = frozenset()",
    ),
    Mutant(
        "a class is the whole identity",
        "return parse(split_frames(identity)[1]).terms[0].tag",
        "return identity",
    ),
    Mutant(
        "checkpoints fold per store",
        'return witness._write(None, "checkpoint", ctx, call, finish)',
        'return witness._write(f"checkpoint:{id(ctx)}", "checkpoint", ctx, call, finish)',
    ),
    Mutant("hits read as unreached", 'return "reached, never wrote"', 'return "unwitnessed"'),
    Mutant("xdist allowed", '"-n",\n        "0",\n', ""),
    Mutant("an empty run reported", "        return 2\n", "        pass\n"),
)


def pins(source: Path | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if source is not None:
        env["EFFECT_WITNESS_SOURCE"] = str(source.relative_to(ROOT))
    return subprocess.run(
        [sys.executable, "-m", "pytest", PINS, "-q", "--no-cov", "-p", "no:randomly", "-x"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    if (baseline := pins(None)).returncode != 0:
        print(baseline.stdout[-2000:])
        print("the unmutated pins fail; no mutant result means anything", file=sys.stderr)
        return 1
    source = SCRIPT.read_text()
    MUTANT.parent.mkdir(parents=True, exist_ok=True)
    survived: list[str] = []
    try:
        for mutant in MUTANTS:
            if (count := source.count(mutant.old)) != 1:
                raise SystemExit(
                    f"mutant {mutant.name!r}: its anchor occurs {count} times, not once"
                )
            MUTANT.write_text(source.replace(mutant.old, mutant.new))
            match pins(MUTANT).returncode:
                case 1:  # a pin failed: the mutant was caught
                    print(f"killed   {mutant.name}")
                case 0:
                    print(f"SURVIVED {mutant.name}")
                    survived.append(mutant.name)
                case code:  # collection or usage error: the mutant never reached a pin
                    print(f"ERROR    {mutant.name}: pytest exited {code}")
                    survived.append(mutant.name)
    finally:
        MUTANT.unlink(missing_ok=True)
    print(f"\n{len(MUTANTS) - len(survived)} of {len(MUTANTS)} killed")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
