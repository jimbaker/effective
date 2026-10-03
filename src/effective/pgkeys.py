"""The psycopg half of `Key`'s serde — one definition, imported by every psycopg consumer.

`effective.keys.Key` names exactly two positions that may treat a key as text: the composer's
splice, and **serde registered against a driver** so no call site ever converts on the way to a
database. sqlite3's half lives in `effective.sqlite` (which owns that driver). This is psycopg's.

**Why its own module rather than a few lines in one of its consumers.** A `psycopg.adapters`
registration is process-global but only takes effect if the module holding it is *imported*, and
the psycopg consumers here do not import each other: `effective.ledger` is SQLAlchemy/SQLModel
(and the core is deliberately DB-free, so the durable handler must not pull it in),
`effective.handlers.absurd` drives the Absurd SDK, `effective.parked` is a read-only fleet
reader. Putting the dumper in any one of them leaves the others unprotected — which is not
hypothetical: it first lived in `handlers.absurd`, and `PostgresLedger.append` promptly failed
with ``cannot adapt type 'Key'`` because the SQLAlchemy path never imports that module.

So: one module, no dependencies beyond psycopg and the leaf `keys`, imported for its side effect
by each consumer. Cheap to import, impossible to half-apply.

There is deliberately **no loader back**. A key read out of a column is text and re-enters the
typed world through `Key.parse`, the one named read boundary — the same asymmetry the Pydantic
schema states (`is_instance` for writers, parse for readers).
"""

import psycopg
import psycopg.adapt

from effective.keys import Key


class KeyDumper(psycopg.adapt.Dumper):
    """Bind a `Key` as its durable text — `stored()`, never `display()`.

    The distinction matters here more than anywhere: `display()` is allowed to be elided or
    truncated for a projection, and this value goes into `UNIQUE(event_id)` on an append-only
    table. A prettified key committed to the canonical record is not repairable.
    """

    def dump(self, obj: Key) -> bytes:
        return obj.stored().encode()


psycopg.adapters.register_dumper(Key, KeyDumper)
