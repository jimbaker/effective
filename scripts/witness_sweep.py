"""Measure the admission witness against what a checkpoint keeps, over a constructed value space.

A race branch records a digest of the values it was handed. The attempt that ran the op is handed
the live value; the attempt that retries is handed what the store gives back. So the witness owes
the store two things, and this sweep measures both:

| property | what it says | what its failure costs |
|---|---|---|
| stability | a value and its round trip witness alike | a retry given the same input fails |
| fidelity | values the store keeps apart witness apart | a retry given another input completes |

Stability is the checkable one, and it implies the half of fidelity that matters: where the store
merges, the witness merges too. What is left is the coarsening the ruling permits, which this
sweep enumerates rather than argues, one class per row.

    uv run python scripts/witness_sweep.py
    uv run python scripts/witness_sweep.py --dsn postgresql://effective:effective@localhost:5432/effective

`wiki/concepts/witness-quotient.md` states what the classes mean.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from effective.handlers.admission import observed
from effective.handlers.durable import _load

DEFAULT_DSN = "postgresql://effective:effective@localhost:5432/effective"


class Held(BaseModel):
    """A model whose one field takes whatever a leaf is."""

    held: Any = None


@dataclass
class Borne:
    """A dataclass whose one field takes whatever a leaf is."""

    borne: Any = None


class Grade(IntEnum):
    ONE = 1


class Tag(StrEnum):
    A = "a"


class Sized(int):
    pass


class Named(str):
    pass


def leaves() -> dict[str, Any]:
    """The scalars a branch can be handed, chosen so that each pair the store might merge or keep
    apart is in the space by construction rather than by a reviewer's recall."""
    return {
        "none": None,
        "true": True,
        "false": False,
        "int-0": 0,
        "int-1": 1,
        "int-minus-1": -1,
        "int-2p53": 2**53,
        "int-2p63": 2**63,
        "int-30-digits": 10**30,
        "float-0": 0.0,
        "float-minus-0": -0.0,
        "float-1": 1.0,
        "float-1e16": 1e16,
        "float-0.1": 0.1,
        "float-denormal": 5e-324,
        "float-max": 1.7976931348623157e308,
        "decimal-1": Decimal("1"),
        "decimal-1.0": Decimal("1.0"),
        "decimal-1.21": Decimal("1.21"),
        "decimal-1.24": Decimal("1.24"),
        "decimal-1e16": Decimal("1E+16"),
        "decimal-30-digits": Decimal(10**30),
        "decimal-0.1": Decimal("0.1"),
        "str-empty": "",
        "str-a": "a",
        "str-1": "1",
        "str-true": "true",
        "str-nfc": unicodedata.normalize("NFC", "é"),
        "str-nfd": unicodedata.normalize("NFD", "é"),
        "str-escape": 'a"\\b\n',
        "str-300": "k" * 300,
        "str-subclass": Named("a"),
        "int-subclass": Sized(1),
        "enum-int": Grade.ONE,
        "enum-str": Tag.A,
        "datetime-naive": datetime(2026, 1, 1, 12, 0),
        "datetime-utc": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        "date": date(2026, 1, 1),
        "time": time(12, 0),
        "uuid": UUID("00000000-0000-4000-8000-000000000001"),
        "bytes-text": b"a",
        "bytes-binary": b"\xff\xfe",
        "str-nul": "a\x00b",
    }


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def placements() -> dict[str, Any]:
    """Where a leaf can sit. A witness that follows the encoding must answer the same at every
    depth, so the sweep varies the position as well as the value."""
    return {
        "bare": lambda leaf: leaf,
        "list": lambda leaf: [leaf],
        "tuple": lambda leaf: (leaf,),
        "dict-value": lambda leaf: {"k": leaf},
        "nested": lambda leaf: {"a": [{"b": leaf}]},
        "model-field": lambda leaf: Held(held=leaf),
        "dataclass-field": lambda leaf: Borne(borne=leaf),
        "dict-written-a-first": lambda leaf: {"a": leaf, "bb": 1, "c" * 40: 2},
        "dict-written-a-last": lambda leaf: {"c" * 40: 2, "bb": 1, "a": leaf},
        "set": lambda leaf: {leaf},
        "set-of-three": lambda leaf: {leaf, "~one", "~two"},
        "frozenset": lambda leaf: frozenset({leaf}),
        "dict-key": lambda leaf: {leaf: 1},
        "set-in-model": lambda leaf: Held(held={leaf, "~one", "~two"}),
        "set-in-dataclass": lambda leaf: Borne(borne={leaf, "~one", "~two"}),
    }


NEEDS_HASHABLE = frozenset(
    {"set", "set-of-three", "frozenset", "dict-key", "set-in-model", "set-in-dataclass"}
)


@dataclass(frozen=True)
class Specimen:
    """One value in the space, named by the leaf it holds and where the leaf sits."""

    leaf: str
    placement: str
    value: Any
    schema: Any = object

    @property
    def name(self) -> str:
        return f"{self.leaf}/{self.placement}"


def reconstructed() -> list[Specimen]:
    """Values whose op declares a type the store's JSON cannot carry, so the schema puts the kind
    back on the way out. A set is the one that matters: the encoding lists it in whatever order
    this process's hash seed gave, and only the declared type makes it a set again."""
    return [
        Specimen("set-of-strings", "typed", {f"k{n}" for n in range(12)}, set[str]),
        Specimen("frozenset-of-ints", "typed", frozenset(range(12)), frozenset[int]),
        Specimen("tuples-in-a-list", "typed", [(1, "a"), (2, "b")], list[tuple[int, str]]),
        Specimen("a-typed-mapping", "typed", {"bb": 1, "a": 2}, dict[str, int]),
        Specimen("a-model", "typed", Held(held={"bb": 1, "a": 2}), Held),
        Specimen("a-set-under-Any", "typed", Held(held={f"k{n}" for n in range(12)}), Held),
    ]


def specimens() -> list[Specimen]:
    """The space: every leaf in every placement its type allows, then the reconstructed cohort."""
    where = placements()
    return [
        Specimen(leaf, placement, build(value))
        for leaf, value in leaves().items()
        for placement, build in where.items()
        if placement not in NEEDS_HASHABLE or _hashable(value)
    ] + reconstructed()


@dataclass(frozen=True)
class Row:
    """One specimen, measured: what the store keeps of it, what the codec makes of it live and of
    what the store gives back, and which of the four outcomes it reached."""

    specimen: Specimen
    outcome: str
    stored: str | None = None
    live: str | None = None
    back: str | None = None
    detail: str = ""


def _witness(value: Any) -> str | None:
    """The witness the handler records for a value a branch returned, as comparable text."""
    taken = observed(["value", value])
    return None if taken is None else json.dumps(taken, separators=(",", ":"))


class Store:
    """One engine's checkpoint round trip: the text it keeps of an encoded value.

    SQLite holds it as JSON text, which keeps an object's keys in the order they were written."""

    name = "sqlite"

    def kept(self, encoded: Any) -> str:
        return json.dumps(encoded)


class Postgres(Store):
    """Absurd's, where a checkpoint is `jsonb`: keys sorted, a zero's sign dropped, `1e+16` read
    back as an integer."""

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)

    def kept(self, encoded: Any) -> str:
        text = json.dumps(encoded)
        with self.conn.cursor() as cur:
            cur.execute(t"SELECT {text}::jsonb::text")
            row = cur.fetchone()
        assert row is not None  # one expression, so the select always answers
        return str(row[0])

    def close(self) -> None:
        self.conn.close()


def measure(store: Store, specimen: Specimen) -> Row:
    """One specimen against one engine, along the path the handler takes.

    A branch is never handed the live domain value: a step returns `_load(schema, raw)` on the
    attempt that ran it as well as on the one that replays it, so the only difference between the
    two witnesses is what the store did to `raw` in between. That is the whole subject, and a
    sweep that witnessed the live value instead would be measuring a call path nothing takes.

    | outcome | reached when |
    |---|---|
    | `unstorable` | the checkpoint encoding refuses the value, so no op could record it |
    | `refused` | the store refuses the encoding, which is the same failure one layer down |
    | `unwitnessed` | the codec takes no witness, so the digest is `None` and the retry fails |
    | `stable` | the attempt that ran the op and the one that replays it witness alike |
    | `unstable` | they do not, so a retry handed the same input fails |
    """
    try:
        encoded = to_jsonable_python(specimen.value)
    except Exception as unencodable:
        return Row(specimen, "unstorable", detail=type(unencodable).__name__)
    try:
        kept = store.kept(encoded)
    except Exception as refused:
        return Row(specimen, "refused", detail=type(refused).__name__)
    ran = _load(specimen.schema, encoded)
    replayed = _load(specimen.schema, json.loads(kept))
    live, back = _witness(ran), _witness(replayed)
    if live is None or back is None:
        return Row(specimen, "unwitnessed", kept, live, back)
    return Row(specimen, "stable" if live == back else "unstable", kept, live, back)


def sweep(store: Store) -> list[Row]:
    return [measure(store, specimen) for specimen in specimens()]


@dataclass(frozen=True)
class Coarsening:
    """One class the codec merges and the store keeps apart: what it merges along one axis, the
    distinct forms the store kept, and where along the other axis it holds."""

    merged: frozenset[str]
    kept: tuple[str, ...]
    holding: tuple[str, ...]


def coarsenings(rows: list[Row], *, over: str) -> list[Coarsening]:
    """Where the witness is coarser than the store, along one axis of the space.

    The space has two, and the relation reads differently along each. `over="leaf"` varies the
    value inside a fixed position, which is the quotient on values the ruling is about.
    `over="placement"` varies the position of a fixed value, which is where the encoding's own
    merges show: a tuple and a list, a set and the array it becomes, a mapping written in two
    orders. A class holding at some positions and not others is the interesting shape, so each
    carries where it holds."""
    varies: Callable[[Specimen], str] = (
        (lambda specimen: specimen.leaf)
        if over == "leaf"
        else (lambda specimen: specimen.placement)
    )
    fixed: Callable[[Specimen], str] = (
        (lambda specimen: specimen.placement)
        if over == "leaf"
        else (lambda specimen: specimen.leaf)
    )
    seen: dict[frozenset[str], dict[str, tuple[str, ...]]] = defaultdict(dict)
    for cohort, within in _cohorts(rows, fixed).items():
        merged: dict[str, set[str]] = defaultdict(set)
        kept: dict[str, set[str]] = defaultdict(set)
        for row in within:
            assert row.live is not None
            assert row.stored is not None
            merged[row.live].add(varies(row.specimen))
            kept[row.live].add(row.stored)
        for witness, members in merged.items():
            if len(kept[witness]) > 1:
                seen[frozenset(members)][cohort] = tuple(sorted(kept[witness]))
    return [
        Coarsening(members, next(iter(where.values())), tuple(sorted(where)))
        for members, where in seen.items()
    ]


def _cohorts(rows: list[Row], by: Callable[[Specimen], str]) -> dict[str, list[Row]]:
    """The measured rows, grouped by the axis held fixed. Only rows a store kept take part in a
    comparison: the rest never reached a checkpoint."""
    within: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.outcome in ("stable", "unstable"):
            within[by(row.specimen)].append(row)
    return within


def refinements(rows: list[Row]) -> list[tuple[str, str, str]]:
    """Where the witness is finer than the store: one stored form under two witnesses, which is
    the failure stability forbids. Reported as its own count, since a sweep that finds one here
    and none among the unstable rows would mean the two measurements disagree."""
    by_stored: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        if row.outcome in ("stable", "unstable") and row.stored is not None:
            assert row.live is not None
            by_stored[row.stored].setdefault(row.live, row.specimen.name)
    return [
        (stored, first, second)
        for stored, seen in by_stored.items()
        if len(seen) > 1
        for first, second in [tuple(sorted(seen.values()))[:2]]
    ]


def _other(over: str) -> str:
    return "placement" if over == "leaf" else "leaf"


def _short(form: str, width: int = 60) -> str:
    return form if len(form) <= width else f"{form[:width]}…"


def report(name: str, rows: list[Row]) -> int:
    """Print one engine's sweep and answer how many rows failed stability."""
    counted: dict[str, int] = defaultdict(int)
    for row in rows:
        counted[row.outcome] += 1
    print(f"\n## {name}: {len(rows)} specimens")
    for outcome in ("stable", "unstable", "unwitnessed", "refused", "unstorable"):
        print(f"  {outcome:12} {counted[outcome]}")
    for outcome in ("unstable", "unwitnessed", "refused", "unstorable"):
        named = [row.specimen.name for row in rows if row.outcome == outcome]
        if named:
            print(f"  {outcome}: {', '.join(sorted(named))}")
    for over, axis in (
        ("leaf", "values, at one position"),
        ("placement", "positions, of one value"),
    ):
        classes = coarsenings(rows, over=over)
        everywhere = set(
            _cohorts(rows, lambda specimen, over=over: getattr(specimen, _other(over)))
        )
        print(
            f"\n  {len(classes)} classes the codec merges and the store keeps apart, over {axis}"
        )
        for held in sorted(classes, key=lambda held: sorted(held.merged)):
            missing = sorted(everywhere - set(held.holding))
            where = "everywhere" if not missing else f"except at {', '.join(missing)}"
            print(f"    {', '.join(sorted(held.merged))}")
            print(f"      kept as {' | '.join(_short(form) for form in held.kept)}, {where}")
    for stored, first, second in refinements(rows):
        print(f"  FINER THAN THE STORE: {stored} witnessed apart by {first} and {second}")
    return counted["unstable"] + len(refinements(rows))


def main(argv: list[str] | None = None) -> int:
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("--dsn", default=DEFAULT_DSN, help="the Postgres to measure jsonb against")
    parsed.add_argument("--sqlite-only", action="store_true", help="skip the Postgres arm")
    args = parsed.parse_args(argv)

    failures = report("sqlite", sweep(Store()))
    if not args.sqlite_only:
        postgres = Postgres(args.dsn)
        try:
            failures += report("postgres", sweep(postgres))
        finally:
            postgres.close()
    print(f"\n{failures} rows where the witness did not survive a round trip")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
