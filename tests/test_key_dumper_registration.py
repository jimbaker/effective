"""A process that builds its Absurd app through the worker binds a `Key` without the ledger.

The suite cannot see this on its own: collecting any module that imports `effective.ledger`
registers the dumper for the whole session, so the probe runs in a process that cannot import it.
"""

import subprocess
import sys
import textwrap

PROBE = textwrap.dedent(
    """
    import sys

    class NoLedger:
        def find_spec(self, name, path=None, target=None):
            if name == "effective.ledger":
                raise ImportError("the ledger is absent")

    sys.meta_path.insert(0, NoLedger())
    import effective.absurd_worker

    import psycopg
    from psycopg.adapt import PyFormat
    from effective.keys import Key

    psycopg.adapters.get_dumper(Key, PyFormat.AUTO)
    """
)


def test_loading_the_worker_registers_a_key_dumper_before_any_connection():
    ran = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr
