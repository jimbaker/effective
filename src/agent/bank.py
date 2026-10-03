"""The banked contrast corpus — what makes a store CURRENT, and how a machine says so.

`tests/test_banked_corpus.py` is the repository's only cross-version reader check: it opens stores
written by OLDER revisions and asserts today's reader still reads them. That fixture is only as
good as the corpus underneath it, and the corpus lives in a local, gitignored directory, so it is
whole on the machine where the campaign ran and absent or half-present anywhere else. A fixture
that fails because of where you are standing teaches nothing about the code.

This module is the corpus's own description, read by three callers that have to agree:

- a corpus writer and gate, which live outside this repository;
- `tests/test_banked_corpus.py`, which skips the halves the local corpus cannot support;
- a human bumping `CURRENT_ERA` when the key grammar moves again, which is the whole maintenance
  procedure: the old tree becomes legacy where it stands, and the gate banks the new one.

**The split that makes this work is reproducibility, and it is not symmetric.** A CURRENT-era
corpus is reproducible (32 trials, about half a cent, one command), so its absence is a condition
the gate REPAIRS. A legacy corpus is not reproducible by any command, because the writer that
wrote it is gone; so its absence is a fact about this machine that the fixture STATES and skips.
Conflating the two is what made a stale checkout look like a broken reader.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCHES = REPO / "benches"
CONTRAST = BENCHES / "contrast"

STAMP_NAME = "STAMP.json"
"""Written at the ROOT of an era tree by the corpus writer. Its presence distinguishes a tree
the writer banked from a directory someone happened to create with the right name."""

CURRENT_ERA = "era-5"
"""Stores written by the CURRENT writer live under this name. Eras are told apart by PATH rather
than by sniffing the keys: the point of a fixture is to state what it expects, and a classifier
that read the keys would agree with any corpus by construction.

**What a reader needs is the key shape each tree on disk holds**, since that is what says whether
a tree replays here. Only the current era does; everything under an older name is READABLE and not
replayable, which is the one-directional guarantee the fixture holds.

    earliest        `react:{scope}{tag}`            a scope spliced into the name
    writer          `task:{id}/react:{tag}`         a frame joined with `/`
    `era-2`         `task:{id};react:{tag}`         a frame as a leading term
    `era-3`         `task:{id};step;react:0`        every key opens with its arm
    `era-4`         `task:{id};step;react:turn,0`   a turn coordinate declares its role
    `era-5`         `task:{id};d:0;step;react:turn` the turn is a `descend` level's frame

**Re-banking is the routine move**: a campaign is 32 trials
and well under a cent, so a writer change re-banks rather than acquiring a translation layer:
migration costs less than the machinery that would avoid it."""

BENCH_SEED = 7
"""The generator seed a bank runs under (`scripts/contrast_run.py --seed`, default 7). Tasks are
regenerated from it rather than stored, so a store banked under another seed is skipped rather
than failed — it is a different corpus, not a broken one."""

BANK_ARMS = ("combinator",)
"""What the corpus writer writes, and deliberately narrower than `ARMS`. The replay pin needs the
`combinator` arm alone — its replay entry point takes only the task and its rows, where the
structured arm additionally needs the harness inputs — and one arm exercises the same
record/replay path. Banking four arms would cost four times as much to pin the same thing."""

BANK_SIZES = ("S", "L")
BANK_ATTEMPTS = 2
BANK_MODEL = "gpt-5-nano"
BANK_BUDGET_USD = 1.00
"""A ceiling, not an estimate: a full combinator bank measured ~$0.005 (32 trials at ~$0.00013).
The gap is deliberate — the ceiling exists to stop a runaway, not to predict the bill."""


@dataclass(frozen=True)
class Stamp:
    """What a bank records about itself, so a later run can tell whether it is still the one asked
    for. Every field is something a mismatch should re-bank over."""

    era: str
    seed: int
    arms: tuple[str, ...]
    sizes: tuple[str, ...]
    attempts: int
    model: str
    revision: str
    banked_at: str
    stores: int
    trials: int


@dataclass(frozen=True)
class CorpusStatus:
    """The gate's whole finding. `problems` empty means the current era is usable; `legacy_stores`
    is reported and never demanded, since nothing can produce one."""

    era: str
    root: Path
    stamp: Stamp | None
    stores: int
    arm_trials: int
    legacy_stores: int
    problems: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.problems


def era_root(era: str = CURRENT_ERA, contrast: Path = CONTRAST) -> Path:
    return contrast / era


def is_current(store: Path, era: str = CURRENT_ERA) -> bool:
    """Classification by path — the rule the fixture and the gate must not disagree about."""
    return era in store.parts


def all_stores(benches: Path = BENCHES) -> list[Path]:
    return sorted(benches.rglob("task.db")) if benches.exists() else []


def trial_files(root: Path, arm: str | None = None) -> list[Path]:
    if not root.exists():
        return []
    found = sorted(root.rglob("trial.json"))
    if arm is None:
        return found
    return [p for p in found if json.loads(p.read_text()).get("arm") == arm]


def read_stamp(root: Path) -> Stamp | None:
    """Absent or unreadable both read as "no stamp" — the caller's repair is the same either way,
    and a half-written stamp is exactly the case where refusing to guess is cheapest."""
    path = root / STAMP_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return None
    known = {f.name for f in fields(Stamp)}
    if not known <= data.keys():
        return None
    kept = {k: v for k, v in data.items() if k in known}
    kept["arms"] = tuple(kept["arms"])
    kept["sizes"] = tuple(kept["sizes"])
    return Stamp(**kept)


def write_stamp(root: Path, stamp: Stamp) -> Path:
    path = root / STAMP_NAME
    path.write_text(json.dumps(asdict(stamp), indent=1) + "\n")
    return path


def status(
    era: str = CURRENT_ERA, contrast: Path = CONTRAST, benches: Path = BENCHES
) -> CorpusStatus:
    """Decide whether this machine holds a corpus the fixture can be pinned against.

    THE DOMAIN, written down because a gate's grammar is its domain and the sentence describing it
    will otherwise quietly claim more. Checked: the era tree exists, carries a stamp naming this
    era and this seed, holds as many stores as the stamp claims, and holds at least one trial for
    the banked arm. NOT checked: that any key parses, that any store replays, that a legacy era
    exists. The first two are `tests/test_banked_corpus.py`'s job — running that module is what
    proves the corpus, and this gate exists only to keep it from skipping vacuously.
    """
    root = era_root(era, contrast)
    stamp = read_stamp(root)
    stores = [p for p in all_stores(benches) if is_current(p, era)]
    arm = BANK_ARMS[0]
    arm_trials = trial_files(root, arm)
    legacy = [p for p in all_stores(benches) if not is_current(p, era)]

    problems: list[str] = []
    if not root.exists():
        problems.append(f"no {era} tree at {root}")
    elif stamp is None:
        problems.append(f"{root / STAMP_NAME} missing or unreadable: not banked by the writer")
    else:
        if stamp.era != era:
            problems.append(f"stamp says era {stamp.era!r}, current era is {era!r}")
        if stamp.seed != BENCH_SEED:
            problems.append(f"stamp seed {stamp.seed} != pinned seed {BENCH_SEED}")
        if stamp.stores != len(stores):
            problems.append(f"stamp banked {stamp.stores} stores, found {len(stores)}")
    if not stores:
        problems.append("no current-era stores — the replayable half of the fixture is vacuous")
    elif not arm_trials:
        problems.append(f"no {arm} trial.json — the replay pin would be vacuous")

    return CorpusStatus(
        era=era,
        root=root,
        stamp=stamp,
        stores=len(stores),
        arm_trials=len(arm_trials),
        legacy_stores=len(legacy),
        problems=tuple(problems),
    )
