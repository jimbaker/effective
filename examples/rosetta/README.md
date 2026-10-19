# Rosetta

One retrieval program, ported to Effective, DSPy, Pydantic AI and LangGraph, and measured on the
questions where the frameworks differ: a refused answer, a crash and its resume, a human approval,
a fork, and the size of each port.

The program searches for passages and asks a model for `reasoning`, `answer` and a `citation`
that must index a passage. Every model and the search tool are offline fakes that count their
calls (`fakes.py`), so the probes run with no network and no keys.

This is its own uv project with its own lock, so none of the three frameworks enters the root
environment. Run the pinned measurements, then print the table:

```sh
uv run --project examples/rosetta pytest examples/rosetta/tests -q
uv run --project examples/rosetta python examples/rosetta/compare.py
```

| file | holds |
|---|---|
| `effective_rag.py`, `dspy_rag.py`, `pydantic_ai_rag.py`, `langgraph_rag.py` | the ports |
| `fakes.py` | the counting search tool and model, and each framework's adapter to them |
| `probes.py` | one probe per question and framework, shared by the tests and `compare.py` |

The first run creates `examples/rosetta/.venv`, which git ignores and the root lint skips.
