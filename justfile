# effective: dev recipes. Run `just` to list.
set dotenv-load
# Pass recipe args as real positional args ("$@") so quoting survives.
set positional-arguments

# A tree that holds more than this repository ships adds its recipes here.
import? 'private.just'

DATABASE_URL := env_var_or_default("DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective")
PG_URL := "${DATABASE_URL:-postgresql://effective:effective@localhost:${PGTEST_PORT:-5432}/effective}"

# A pinned test-ordering seed so the gates (`test`, `check`, `cov`) are deterministic.
# pytest-randomly still shuffles, at this seed. Shake on demand with `GATE_SEED=$RANDOM just cov`.
GATE_SEED := env_var_or_default("GATE_SEED", "0")

# Source files holding channel templates, subject to the channel lint.
CHANNEL_SRCS := "src/effective/cache.py src/effective/react.py src/effective/interpreters/cli.py src/effective/interpreters/openai.py src/effective/interpreters/precise_edit.py src/agent/skillsbench.py src/examples/coder/tools.py src/examples/coder/machine.py src/examples/deep_research/research.py src/examples/startup/asking.py src/examples/startup/incident.py src/examples/startup/voice.py examples/first_workflow.py"

# List recipes
default:
    @just --list

# --- dev loop ---

# Install / sync dependencies
install:
    uv sync

# Full test suite, seed-pinned. Postgres-backed tests skip without a database; `just pgt-up`
# first to run them, on `PGTEST_PORT` if set. Each xdist worker gets its own copy of a template
# database.
test: key-registry
    DATABASE_URL="{{PG_URL}}" uv run pytest -q -n auto --dist loadfile --randomly-seed={{GATE_SEED}}

# Infra-free tests only: a dead DATABASE_URL makes every backend-gated test skip.
test-fast: key-registry
    DATABASE_URL="postgresql://dead:dead@127.0.0.1:1/dead" uv run pytest -q -n auto --no-cov

# The durable lane alone, with Postgres required rather than optional.
pgt-test:
    EFFECTIVE_REQUIRE_PG=1 DATABASE_URL="{{PG_URL}}" uv run pytest tests/test_conformance.py tests/test_absurd_integration.py tests/test_replay_crash.py tests/test_replay_improve.py tests/test_replay_permission.py tests/test_spawned_subagent.py tests/test_skills_durable.py tests/test_absurd_seed_reader.py tests/test_parked_reader.py tests/test_durable_voi_bridge_absurd.py tests/test_fork_sweep.py tests/test_fork_sweep_absurd.py tests/test_carrier_audit.py tests/test_respawn_hazards_pending.py tests/test_durable_sleep_and_drain_depth.py tests/test_grant_aliasing.py tests/test_await_occurrence_walks.py tests/test_placement_visible_to_domain_layer.py tests/test_span_key_joins_the_tape.py tests/test_machine_telemetry_joins_the_tape.py tests/test_worker_death.py tests/test_settle.py tests/test_race_durable.py -v

# Lint: ruff, the effect-boundary and key rules, the f-string gate, and ty.
lint:
    uv run ruff check src tests examples scripts
    uv run python -m effective.lint
    uv run python -m effective.lint --layers
    uv run python -m effective.lint --channels {{CHANNEL_SRCS}}
    uv run python -m effective.lint --deps src/effective src/agent src/tui src/examples
    uv run python -m effective.lint --ledger-reads src/effective src/agent scripts
    uv run python -m effective.lint --sql-templates src/effective src/agent src/examples
    uv run python -m effective.lint --totality src
    uv run python -m effective.lint --ordered-arms src
    uv run python -m effective.lint --totality-escapes src
    uv run python -m effective.lint --key-composition src/effective src/agent src/tui src/examples scripts
    uv run python -m effective.lint --terminal-holes tests
    uv run python -m effective.lint --authority-tags src/effective src/agent src/examples examples
    uv run python -m effective.lint --authority-scopes src/effective src/agent src/examples examples
    uv run python -m effective.lint --key-registry
    uv run python -m effective.lint --coordinate-roles src/effective src/agent src/examples examples
    uv run python -m effective.lint --python-codegen src tests examples scripts
    uv run python -m effective.lint --key-borrowing tests
    uv run python -m effective.lint --key-literals src
    uv run python -m effective.lint --forged-join src
    # An f-string belongs in a t-string processor's render backend and in building an exception.
    # Every other one is recorded with the processor it wants; the gate fails on a new one and on
    # a record nothing produces, so the record can only shrink.
    uv run python scripts/fstring_sweep.py --gate src
    uv run python -m effective.lint --lazy-imports src/effective src/agent src/tui src/examples
    uv run python -m effective.lint --working-notes src/effective src/agent src/tui src/examples tests scripts examples
    uv run python -m effective.lint --public-prose src/effective src/agent
    uv run python -m effective.lint --role-coverage src examples scripts
    uv run python -m effective.lint --layer-coverage src examples scripts
    uv run python -m effective.lint --test-citations src tests scripts docs examples wiki
    # ty is pinned: `uvx ty` would otherwise resolve the newest release at each invocation.
    uvx ty@0.0.73 check --python .venv

# Type-check alone
typecheck:
    uvx ty@0.0.73 check --python .venv

# Format and autofix
fmt:
    uv run ruff format src tests examples scripts
    uv run ruff check --fix src tests examples scripts

# Dangling links in docs/ and wiki/
docs-check *ARGS:
    uv run python scripts/link_check.py "$@"

# Wiki [[links]] and orphan pages
wiki-lint *ARGS:
    uv run python scripts/wiki_lint.py "$@"

# Key-shape sweep, gated against its baseline
key-check *ARGS:
    uv run python scripts/key_sweep.py --gate "$@"

# Key-shape sweep, as a report
key-sweep *ARGS:
    uv run python scripts/key_sweep.py "$@"

# Write build/key-registry.json, the source map of every key namespace production mints
key-registry:
    uv run python -m effective.lint --key-registry

# Browse the key registry
key-view *ARGS:
    uv run python scripts/key_view.py "$@"

# Mermaid fences: a syntax check, milliseconds (defaults to README.md, docs/, wiki/)
lint-mermaid *ARGS:
    uv run python scripts/lint_mermaid.py "$@"

# Render each Mermaid fence in FILE to PNG under build/mermaid/<stem>/, to look at before
# committing. mmdc runs in the digest-pinned image from infra/mermaid/PIN.txt, offline
mermaid-render FILE WIDTH="1400":
    #!/usr/bin/env bash
    set -euo pipefail
    digest=$(awk '$1 == "digest:" {print $2}' infra/mermaid/PIN.txt)
    image=$(awk '$1 == "image:" {print $2}' infra/mermaid/PIN.txt)
    out="build/mermaid/$(basename "{{FILE}}" .md)"
    rm -rf "$out" && mkdir -p "$out"
    uv run python -c 'import sys, pathlib; from scripts.lint_mermaid import fences; [pathlib.Path(sys.argv[2], f"{i:02d}.mmd").write_text(b) for i, b in enumerate(fences(pathlib.Path(sys.argv[1]).read_text()))]' "{{FILE}}" "$out"
    for f in "$out"/*.mmd; do
      podman run --rm --network=none --userns=keep-id --user "$(id -u):$(id -g)" \
        -v "$PWD/$out":/data:Z -w /data "$image@$digest" \
        -i "$(basename "$f")" -o "$(basename "${f%.mmd}").png" -b white -w {{WIDTH}} >/dev/null
    done
    echo "rendered $(ls "$out"/*.png | wc -l) diagram(s) to $out"

# Validate a span file as OTLP (offline, free)
eval-jsonl FILE:
    uv run python -m effective.telemetry {{FILE}}

# Toolchain preconditions: a green gate is green on the toolchain it ran on
preflight *ARGS:
    uv run python scripts/preflight.py "$@"

# The gate: preflight, lint, docs, wiki, keys, full suite
check: preflight lint docs-check wiki-lint key-check test

# Coverage gate: full suite against the test Postgres, enforcing 90%. The durable paths need a
# real backend to execute, so the number is only meaningful with Postgres up.
cov:
    #!/usr/bin/env bash
    set -eo pipefail
    if podman ps -q --filter name=effective-pg-test | grep -q .; then
        ours=0   # someone else's container: leave it as we found it
    else
        ours=1
    fi
    just pgt-up
    if [ "$ours" = 1 ]; then trap 'just pgt-down' EXIT; fi
    DATABASE_URL="{{PG_URL}}" \
        uv run pytest -q -n auto --dist loadfile --cov-fail-under=90 --randomly-seed={{GATE_SEED}}

# Per-test coverage contexts, for asking which test covers a line
cov-contexts *ARGS:
    #!/usr/bin/env bash
    set -eo pipefail
    if podman ps -q --filter name=effective-pg-test | grep -q .; then
        ours=0
    else
        ours=1
    fi
    just pgt-up
    if [ "$ours" = 1 ]; then trap 'just pgt-down' EXIT; fi
    COVERAGE_FILE=.coverage-contexts COVERAGE_CORE=pytrace \
    DATABASE_URL="{{PG_URL}}" \
        uv run pytest -q --randomly-seed={{GATE_SEED}} \
            --cov=effective --cov=agent --cov-context=test --cov-report= "$@"
    echo "wrote .coverage-contexts; query it with scripts/cov_contexts.py"

# --- examples ---

# The terminal run viewer, on a demo run
tui-demo *args:
    uv run python examples/tui_demo.py "$@"

# The terminal run viewer
tui *args:
    uv run python -m tui "$@"

# The run dashboard demo
dashboard *ARGS:
    uv run python examples/dashboard_demo.py "$@"

# --- graph layout (elkjs, sandboxed) ---

elk-image:
    podman build -t effective-elk:0.12.0 infra/elkjs/

elk-setup:
    #!/usr/bin/env bash
    set -eo pipefail
    cd infra/elkjs && npm ci --ignore-scripts --no-audit --no-fund
    for entry in elk.bundled.js elk-worker.min.js; do
        expect=$(awk -v k="$entry.sha256" '$1==k{print $2}' PIN.txt)
        actual=$(sha256sum "node_modules/elkjs/lib/$entry" | cut -d' ' -f1)
        [ "$expect" = "$actual" ] || { echo "$entry sha256 MISMATCH: $expect != $actual" >&2; exit 1; }
        echo "$entry OK: $actual"
    done

elk-test:
    uv run pytest tests/test_graphlayout.py tests/test_graphlayout_elkjs.py -q --no-cov

elk-demo:
    uv run python -m effective.graphlayout.demo

# --- formal models (Lean + Quint, pinned per machine) ---

# Once per machine: quint via `npm ci`, Apalache fetched and sha256-verified
formal-setup:
    bash scripts/formal_setup.sh

formal-quint:
    bash scripts/formal_quint_container.sh

formal-quint-host:
    #!/usr/bin/env bash
    set -eo pipefail
    QUINT="$PWD/infra/formal/node_modules/.bin/quint"
    [ -x "$QUINT" ] || { echo "no pinned quint: run 'just formal-setup' first" >&2; exit 1; }
    (cd formal/quint && "$QUINT" typecheck effective.qnt && "$QUINT" typecheck gather.qnt \
        && "$QUINT" typecheck budget_confluence.qnt \
        && "$QUINT" typecheck govern_park.qnt && "$QUINT" typecheck race.qnt)
    echo "formal-quint-host: OK"

# Lean build + quint typecheck (seconds)
formal:
    #!/usr/bin/env bash
    set -eo pipefail
    export PATH="$HOME/.elan/bin:$PATH"
    (cd formal/lean && lake build)
    just formal-quint
    echo "formal: lean build + quint typecheck OK"

# Regenerate the conformance vectors the Python tests read
formal-vectors:
    #!/usr/bin/env bash
    set -eo pipefail
    export PATH="$HOME/.elan/bin:$PATH"
    (cd formal/lean && lake build enforce_vectors >/dev/null 2>&1 \
      && lake exe enforce_vectors 2>/dev/null) > formal/enforce_vectors.json
    (cd formal/lean && lake build decide_vectors >/dev/null 2>&1 \
      && lake exe decide_vectors 2>/dev/null) > formal/decide_vectors.json
    (cd formal/lean && lake build govern_vectors >/dev/null 2>&1 \
      && lake exe govern_vectors 2>/dev/null) > formal/govern_vectors.json
    echo "formal-vectors: regenerated formal/{enforce,decide,govern}_vectors.json"

formal-image:
    podman build -t effective-formal:0.32.0-0.56.1 infra/formal/

# Every registered model check, sandboxed (rootless Podman, no network)
formal-verify:
    bash scripts/formal_verify_container.sh

formal-verify-host:
    bash scripts/formal_verify.sh

# --- test infrastructure (rootless Podman) ---

# Start and bootstrap the digest-pinned test Postgres: role, db, Absurd schema, migrations
pgt-up:
    bash scripts/pgtest_up.sh

pgt-down:
    podman rm -f effective-pg-test

pgt-clean:
    DATABASE_URL="{{PG_URL}}" uv run python scripts/pgtest_clean.py

redis-up:
    bash scripts/redistest_up.sh

redis-down:
    podman rm -f effective-redis-test

redis-probe:
    EFFECTIVE_REQUIRE_REDIS=1 REDIS_URL="redis://localhost:${REDISTEST_PORT:-6379}/0" uv run pytest tests/test_redis_streams_probe.py -v --no-cov

# The race suites on a free-threaded CPython, failing if the GIL comes back on
race-free-threaded runs="3":
    uv venv --quiet --allow-existing --python cpython-3.14.7+freethreaded build/venv-ft
    uv export --quiet --no-hashes --format requirements-txt --no-emit-project --no-emit-workspace -o build/ft-requirements.txt
    grep -viE '^(psycopg-binary|ast-grep-py|pydantic-monty|httptools)==|^\./' build/ft-requirements.txt > build/ft-requirements-core.txt
    uv pip install --quiet --python build/venv-ft/bin/python --no-deps --only-binary :all: -r build/ft-requirements-core.txt
    uv pip install --quiet --python build/venv-ft/bin/python --no-deps ./infra/tdom -e .
    for n in $(seq 1 {{runs}}); do PYTHON_GIL=0 PSYCOPG_IMPL=python EFFECTIVE_REQUIRE_PG=1 DATABASE_URL="{{PG_URL}}" build/venv-ft/bin/python scripts/gil_free_pytest.py tests/test_race_admission.py tests/test_race_durable.py tests/test_race.py tests/test_race_deadline.py tests/test_race_shapes.py tests/test_watched_descent.py tests/test_hedge.py tests/test_settle.py tests/test_choice.py tests/test_race_publish_seal.py -q --no-cov -p no:randomly -p no:cacheprovider || exit 1; done

# --- database ---

# Apply the ledger migrations to DATABASE_URL
migrate:
    DATABASE_URL="{{DATABASE_URL}}" uv run alembic upgrade head
