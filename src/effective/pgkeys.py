"""The psycopg half of `Key`'s serde — one definition, imported by every psycopg consumer.

`effective.keys.Key` names exactly two positions that may treat a key as text: the composer's
splice, and **serde registered against a driver** so no call site ever converts on the way to a
database. sqlite3's half lives in `effective.engines.sqlite`, which owns that driver; this is
psycopg's.

**Its own module.** A `psycopg.adapters` registration is process-global, takes effect only once the
module holding it is imported, and reaches only connections opened after it. So each module that
opens a connection a `Key` may reach imports this one for its side effect, before it connects: the
SQLAlchemy ledger (`effective.ledger`) and the Absurd worker. Effective itself hands the Absurd SDK
text on every path, through its ctx adapters, and the durable handler imports neither psycopg nor
this module.

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
