"""The SQLite engine's former path: re-exports the public names of `effective.engines.sqlite`
while consumers outside this repository move (`retire-the-engine-shims-task`).

At run time this name is the defining module itself, so a patch through it reaches the engine."""

import sys

import effective.engines.sqlite as _defining
from effective.engines.sqlite import ClaimLost as ClaimLost
from effective.engines.sqlite import IncompatibleStore as IncompatibleStore
from effective.engines.sqlite import SqliteApp as SqliteApp
from effective.engines.sqlite import SqliteLedger as SqliteLedger
from effective.engines.sqlite import SqliteTaskContext as SqliteTaskContext
from effective.engines.sqlite import TaskSnapshot as TaskSnapshot
from effective.engines.sqlite import connect as connect
from effective.engines.sqlite import enable_wal as enable_wal
from effective.engines.sqlite import unclosed_connections as unclosed_connections

sys.modules[__name__] = _defining
