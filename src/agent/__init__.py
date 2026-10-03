"""agent — the bench/eval half: benchmarks, their tool catalogs, and what measures a run.

The line to `effective` is which side of the effect boundary a module sits on. What yields an op
is substrate (`effective.react`, the combinators); what answers one is an interpreter
(`effective.interpreters`). `agent` keeps what measures runs, and `bracket`, a turn caller with a
decode strategy and a benchmark's 12-tool catalog.

This package re-exports nothing: import from the module that owns the name.
"""
