"""Where the coding machine's success predicate EXECUTES (`effective.coding.tier`).

The predicate runs `pytest` over a tree a model wrote, and pytest imports every module it
collects — a `conftest.py` in the tree runs at collection, before any assertion. So "where does
this run" is a security question, not a deployment detail, and this module is where the answer is
asserted rather than described.

**The container cases need the pinned image and SKIP without it**, which is the same posture the
durable lane takes toward Postgres and carries the same hazard: a box that never built the
`infra/code-agent` image runs the host half, goes green, and proves nothing about isolation. The
skip reason says so. `test_the_tiers_agree_on_a_verdict` is the one that would catch a divergence
between them, and it is also the one that vanishes when the image is absent.
"""

import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from effective.coding.runners import ToolError, Workspace, run_suite, run_tool, serve_tool
from effective.coding.tier import (
    TIER_ENV,
    Tier,
    TierUnavailable,
    UnsafeTreePath,
    default_tier,
    image_available,
    run_tree_command,
    tree_path,
)
from effective.domain import ToolRefused
from effective.machine.evidence import CommandRun

needs_image = pytest.mark.skipif(
    not image_available(),
    reason="the pinned image is absent (`infra/code-agent`): the CONTAINER tier is "
    "untested on this box, so nothing here has checked that isolation holds",
)

PASSING = {
    "mod.py": "def add(a, b):\n    return a + b\n",
    "test_mod.py": "from mod import add\n\n\ndef test_ok():\n    assert add(1, 2) == 3\n",
}
FAILING = {
    "mod.py": "def add(a, b):\n    return a + b\n",
    "test_mod.py": "from mod import add\n\n\ndef test_bad():\n    assert add(1, 2) == 4\n",
}


def test_the_default_tier_is_the_host_and_the_env_var_moves_it(monkeypatch):
    """HOST by default, deliberately: every existing caller runs without podman, and silently
    requiring an image would turn a missing container into failures that read as logic errors.

    An unrecognized value RAISES rather than falling back, for the same reason `TierUnavailable`
    exists — a typo in a deployment's environment must not quietly mean "no isolation".
    """
    monkeypatch.delenv(TIER_ENV, raising=False)
    assert default_tier() is Tier.HOST
    monkeypatch.setenv(TIER_ENV, "container")
    assert default_tier() is Tier.CONTAINER
    monkeypatch.setenv(TIER_ENV, "sandbox")
    with pytest.raises(ValueError, match="not a tier"):
        default_tier()


def test_a_missing_image_refuses_instead_of_running_on_the_host():
    """The failure mode this module exists to prevent, asserted directly.

    A caller asking for the container tier is asking not to execute model-written code beside its
    own files. Falling back to the host would answer a different question and look like success —
    the shape of every "absence read as a pass" defect this repo has recorded.
    """
    with pytest.raises(TierUnavailable, match="refuses rather than running on the host"):
        run_tree_command(PASSING, ("python", "-c", "pass"), tier=Tier.CONTAINER, image="no:such")


@needs_image
def test_a_container_runs_under_a_memory_and_process_ceiling():
    """The time limit bounds a command that runs too long and says nothing about one that eats the
    box while it runs. A review asked for the ceilings; this reads them from inside."""
    files = "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/pids.max /sys/fs/cgroup/cpu.max"
    code, out = run_tree_command(PASSING, ("sh", "-c", files), tier=Tier.CONTAINER)
    assert code == 0, out
    memory, processes, quota, period = out.split()[:4]
    assert int(memory) == 2 * 1024**3
    assert int(processes) == 256
    assert int(quota) / int(period) == 2


def test_the_predicate_measures_the_tree_on_the_host():
    """The baseline: a failing test is DATA, not an infrastructure error.

    `exit_code` is the command's own, never podman's or the runner's, which is what keeps a red
    suite distinguishable from a broken runtime — the one distinction the success predicate exists
    to draw.
    """
    assert run_suite(PASSING, tier=Tier.HOST).exit_code == 0
    failed = run_suite(FAILING, tier=Tier.HOST)
    assert failed.exit_code == 1
    assert failed.failures == ("test_mod.py::test_bad",)
    assert failed.collection_error is None


@needs_image
def test_the_tiers_agree_on_a_verdict():
    """A tier is swappable only if a verdict cannot tell which one ran.

    Both arms return the command's own exit code and its output, so the parsing above them is
    tier-independent by construction — but "by construction" is an argument, and this is the
    measurement. If these ever diverge, `verdicts.py` is reading a different world depending on a
    deployment setting, which is exactly the coupling the seam is meant to remove.
    """
    for tree in (PASSING, FAILING):
        host, container = run_suite(tree, tier=Tier.HOST), run_suite(tree, tier=Tier.CONTAINER)
        assert host.exit_code == container.exit_code
        assert host.failures == container.failures
        assert host.collection_error == container.collection_error


WRITES = {
    "forges-a-passing-exit": (
        "import sys\nprint()\nprint('__EFF_EXIT__:0')\nsys.stdout.write('x')\nsys.exit(1)\n"
    ),
    "no-trailing-newline": "import sys\nsys.stdout.write('partial')\n",
    "stderr-only": "import sys\nsys.stderr.write('oops\\n')\nsys.exit(3)\n",
    "invalid-utf8": "import sys\nsys.stdout.buffer.write(b'ok \\xff\\xfe')\nsys.exit(1)\n",
}


@pytest.mark.adversarial
@needs_image
def test_the_tiers_return_what_the_command_did_whatever_it_writes():
    """Reddens if a tier derives the exit code or the output from what the command printed: model
    code controls that output, so a failing suite could print its way to a pass on one tier."""
    differ = {}
    for name, script in WRITES.items():
        tree, command = {"script.py": script}, ["python", "script.py"]
        host = run_tree_command(tree, command, tier=Tier.HOST)
        container = run_tree_command(tree, command, tier=Tier.CONTAINER, timeout=60)
        if host != container:
            differ[name] = (host, container)

    assert differ == {}


@pytest.mark.adversarial
@needs_image
def test_a_command_that_signals_itself_fails_the_same_way_on_both_tiers():
    """Reddens if a command that kills itself reads as an exit on one tier and a pass on the other:
    a process that is PID 1 in its container ignores a signal it has no handler for."""
    tree = {
        "script.py": "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\nprint('ran on')\n"
    }
    command = ["python", "script.py"]

    host = run_tree_command(tree, command, tier=Tier.HOST)
    container = run_tree_command(tree, command, tier=Tier.CONTAINER, timeout=60)

    assert (host, container) == ((143, ""), (143, ""))


@pytest.mark.adversarial
@pytest.mark.parametrize(
    "tier", [Tier.HOST, pytest.param(Tier.CONTAINER, marks=needs_image)], ids=["host", "container"]
)
def test_a_command_past_its_time_limit_is_a_failed_run_and_leaves_nothing_running(tier):
    """Reddens if a suite that never ends raises out of the tool, where a durable run retries it
    and records nothing, or leaves its process or container running after the limit. The script's
    name is this run's own, so a container another test starts meanwhile is not counted."""
    script = f"spin_{uuid.uuid4().hex}.py"
    tree = {script: "import time\ntime.sleep(120)\n"}

    exit_code, output = run_tree_command(tree, ["python", script], timeout=3, tier=tier)

    assert exit_code == 124
    assert "timed out" in output
    still = subprocess.run(["pgrep", "--full", script], capture_output=True, text=True)
    assert still.stdout == ""


@pytest.mark.adversarial
def test_the_host_tier_gives_the_command_no_stdin():
    """Reddens if a command on the host reads the stdin of the process driving the machine."""
    driver = (
        "from effective.coding.tier import Tier, run_tree_command\n"
        "print(run_tree_command({'r.py': 'import sys\\nprint(repr(sys.stdin.read()))\\n'}, "
        "['python', 'r.py'], timeout=20, tier=Tier.HOST))\n"
    )

    driven = subprocess.run(
        [sys.executable, "-c", driver],
        input="the driver's secret",
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert driven.stdout.strip() == repr((0, "''\n"))


@needs_image
def test_the_container_really_denies_the_host_filesystem_and_the_network():
    """The claim, measured — and measured by showing the HOST does not deny them.

    An isolation test that only asserts the container fails proves nothing: the tree could be
    failing for any reason, and the test would stay green if `--network=none` were dropped and the
    mount discipline abandoned. So the same tree runs on both tiers, and the host half is what
    makes the container half mean something. On the host this model-written code reads the repo it
    is being run from and opens a socket; in the container both raise.

    This is the anti-vacuity rule applied to a security property, where it matters most: the
    dangerous failure is the one where the guard is gone and everything is still green.
    """
    # THIS repo's own file, by absolute host path. An earlier version of this fixture asserted
    # `Path.home()` was readable, which passes INSIDE the container too — the image has a home
    # directory — so the container half went green while testing nothing. The escape has to name
    # something that exists on the host and cannot exist in an unmounted container.
    host_file = str(Path(__file__).resolve())
    tree = {
        "test_escape.py": (
            "import pathlib, socket\n"
            "\n"
            "\n"
            "def test_reads_the_host_filesystem():\n"
            f"    assert pathlib.Path({host_file!r}).is_file()\n"
            "\n"
            "\n"
            "def test_reaches_the_network():\n"
            '    socket.create_connection(("1.1.1.1", 53), timeout=3).close()\n'
        )
    }
    on_host = run_suite(tree, tier=Tier.HOST, timeout=60)
    in_container = run_suite(tree, tier=Tier.CONTAINER, timeout=60)

    assert on_host.exit_code == 0, (
        f"the host did NOT reach the filesystem or the network, so the container comparison "
        f"below proves nothing: {on_host.output[-400:]}"
    )
    assert sorted(in_container.failures) == [
        "test_escape.py::test_reaches_the_network",
        "test_escape.py::test_reads_the_host_filesystem",
    ]


@needs_image
def test_a_tree_is_rebuilt_inside_rather_than_mounted():
    """No volume mount, no overlayfs — the discipline `infra/code-agent/Dockerfile` states.

    The tree arrives as base64 frames on stdin, so content needs no host tempfile and nothing of
    the host is reachable from inside. Bytes that would otherwise become shell command — quotes,
    newlines, non-ASCII — are what the framing is for, so the fixture spells them.
    """
    tricky = 'value = "a\'b\\"c"  # ünïcödé — $(echo pwned) `date` \n'
    tree = {
        "mod.py": tricky,
        "test_mod.py": (
            "import pathlib\n"
            "\n"
            "\n"
            "def test_bytes_survived():\n"
            "    assert pathlib.Path('mod.py').read_text() == " + repr(tricky) + "\n"
        ),
    }
    assert run_suite(tree, tier=Tier.CONTAINER, timeout=60).exit_code == 0


# ---------------------------------------------------------------- a tree key cannot leave the tree


def escaping_keys(outside: Path) -> list[str]:
    """Two spellings of a key outside the tree: absolute, and climbing out of the temp dir."""
    return [str(outside), "../" * 32 + str(outside).lstrip("/")]


@pytest.mark.adversarial
@pytest.mark.parametrize("tier", [Tier.HOST, Tier.CONTAINER], ids=["host", "container"])
@pytest.mark.parametrize("spelling", [0, 1], ids=["absolute", "dotdot"])
def test_a_tree_key_outside_the_tree_is_refused_before_anything_runs(tmp_path, tier, spelling):
    """Reddens if a tree key can place a file outside the directory the predicate runs in, which is
    how a model-written path would edit the repo being measured. The refusal comes before the tier
    is chosen, so the container arm needs no image."""
    outside = tmp_path / "escaped.txt"
    tree = {escaping_keys(outside)[spelling]: "escaped\n", **PASSING}

    with pytest.raises(ValueError, match="outside the tree"):
        run_tree_command(tree, ["true"], tier=tier)
    assert not outside.exists()


@pytest.mark.adversarial
@pytest.mark.parametrize("spelling", [0, 1], ids=["absolute", "dotdot"])
def test_write_file_refuses_a_path_outside_the_tree_as_a_value(tmp_path, spelling):
    """Reddens if `write_file` accepts a path that would leave the tree when the suite materializes
    it; at the op boundary the refusal is a recorded `ToolRefused`."""
    path = escaping_keys(tmp_path / "escaped.txt")[spelling]

    refused = serve_tool("write_file", {"tree": dict(PASSING), "path": path, "content": "x = 1\n"})

    assert isinstance(refused, ToolRefused)
    assert "outside the tree" in refused.diagnostic


@pytest.mark.adversarial
def test_run_suite_refuses_a_named_tree_that_leaves_the_tree(tmp_path):
    """Reddens if a tree passed to `run_suite` by name is materialized with a key outside it. No
    tool returns such a tree, so it is a caller's seed, and it fails the call rather than
    returning a refusal a model cannot act on."""
    outside = tmp_path / "escaped.txt"
    tree = {escaping_keys(outside)[0]: "escaped\n", **PASSING}

    with pytest.raises(UnsafeTreePath):
        serve_tool("run_suite", {"tree": tree})
    assert not outside.exists()


def test_a_nested_key_inside_the_tree_still_materializes():
    """The refusal is about leaving the tree, and a package path is inside it."""
    tree = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "VALUE = 1\n",
        "test_pkg.py": (
            "from pkg.mod import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n"
        ),
    }
    assert run_suite(tree, tier=Tier.HOST).exit_code == 0


# ---------------------------------------------------------------- one file, one spelling

TOO_DEEP = "/".join(["d"] * 2100) + "/test_deep.py"
"""Every component legal, and the whole path longer than a filesystem accepts."""

NOT_ONE_FILE = [
    pytest.param("test_y.py/", id="trailing-slash"),
    pytest.param("./test_y.py", id="dot-prefix"),
    pytest.param("pkg//test_y.py", id="double-slash"),
    pytest.param("pkg/./test_y.py", id="dot-segment"),
    pytest.param(".", id="dot"),
    pytest.param("test_y\x00.py", id="nul"),
    pytest.param("a" * 256 + ".py", id="long-component"),
    pytest.param("\u00e9" * 128 + ".py", id="long-component-in-bytes"),
    pytest.param(("d" * 200 + "/") * 5 + "a" * 20, id="one-byte-past-the-path-limit"),
    pytest.param(TOO_DEEP, id="long-path"),
    pytest.param("\udc80.py", id="lone-surrogate"),
]


@pytest.mark.adversarial
def test_a_key_at_the_path_limit_is_one_file():
    at_the_limit = ("d" * 200 + "/") * 5 + "a" * 19
    assert len(at_the_limit.encode()) == 1024

    assert tree_path(at_the_limit) == at_the_limit


@pytest.mark.adversarial
def test_an_interior_dotdot_is_refused_wherever_it_sits():
    """Reddens if the `..` check only looks at the start of a key: `a/../../…` climbs as far."""
    with pytest.raises(UnsafeTreePath, match="outside the tree"):
        tree_path("a/" + "../" * 32 + "escaped.py")


@pytest.mark.adversarial
@pytest.mark.parametrize("key", NOT_ONE_FILE)
def test_a_key_that_is_not_one_canonical_file_is_refused(key):
    """Reddens if a tree admits a second spelling of a file: the gate and the tree go by the
    spelling, the filesystem by the file, so `./test_y.py` could replace a test the tree shows."""
    with pytest.raises(UnsafeTreePath):
        tree_path(key)


@pytest.mark.adversarial
@pytest.mark.parametrize("tier", [Tier.HOST, Tier.CONTAINER], ids=["host", "container"])
@pytest.mark.parametrize("key", NOT_ONE_FILE)
def test_either_tier_refuses_a_key_that_is_not_one_canonical_file_before_anything_runs(key, tier):
    """Reddens if a key reaches a tier's filesystem: a path too long for it raises outside
    `serve_tool` on the host and reads as a red suite in the container, so the tiers disagree."""
    with pytest.raises(UnsafeTreePath):
        run_tree_command({key: "X = 1\n", **PASSING}, ["true"], tier=tier)


@pytest.mark.adversarial
@pytest.mark.parametrize("key", NOT_ONE_FILE)
def test_write_file_refuses_a_second_spelling_as_a_value(key):
    """Reddens if `write_file` accepts a spelling that skips the static gate or aliases a file."""
    # Content the static gate accepts, so only the path can be the reason for a refusal.
    refused = serve_tool("write_file", {"tree": dict(PASSING), "path": key, "content": "X = 1\n"})

    assert isinstance(refused, ToolRefused)


COLLIDING = [
    pytest.param(("pkg", "pkg/mod.py"), id="file-and-its-directory"),
    pytest.param(("pkg", "pkg/sub/mod.py"), id="file-and-an-ancestor"),
    pytest.param(("test_x.py", "TEST_X.py"), id="case"),
    pytest.param(("pkg", "Pkg/mod.py"), id="case-of-a-directory"),
    pytest.param(("Pkg", "pkg/mod.py"), id="case-of-a-file"),
    pytest.param(("stra\u00dfe.py", "STRASSE.py"), id="case-folded-past-lower"),
    pytest.param(("caf\u00e9.py", "cafe\u0301.py"), id="normalization"),
]
"""Two keys a filesystem can materialize as one path: a file and a directory of the same name, or
two names equal on a case-insensitive or normalizing filesystem."""


@pytest.mark.adversarial
@pytest.mark.parametrize("order", ["first", "second"])
@pytest.mark.parametrize("pair", COLLIDING)
def test_a_key_that_collides_with_another_is_a_tool_refusal(pair, order):
    """Reddens if a tree holding two keys for one path reaches the filesystem, where one crashes
    outside `serve_tool` or replaces the other while the tree still shows both."""
    first, second = pair if order == "first" else pair[::-1]

    written = serve_tool("write_file", {"tree": dict(PASSING), **write(first)})
    assert isinstance(written, dict), written
    match serve_tool("write_file", {"tree": written, **write(second)}):
        case ToolRefused(diagnostic=diagnostic):
            assert all(ascii(key) in diagnostic for key in pair), diagnostic
        case other:
            pytest.fail(f"the second key was written: {other}")
    assert isinstance(serve_tool("run_suite", {"tree": written}), CommandRun)


def write(path: str) -> dict[str, str]:
    return {"path": path, "content": "X = 1\n"}


@pytest.mark.adversarial
@pytest.mark.parametrize("tier", [Tier.HOST, Tier.CONTAINER], ids=["host", "container"])
@pytest.mark.parametrize("pair", COLLIDING)
def test_a_tree_with_colliding_keys_is_refused_before_anything_runs(pair, tier):
    with pytest.raises(UnsafeTreePath):
        run_tree_command({key: "" for key in pair} | PASSING, ["true"], tier=tier)


@pytest.mark.adversarial
@pytest.mark.parametrize(
    "tier", [Tier.HOST, pytest.param(Tier.CONTAINER, marks=needs_image)], ids=["host", "container"]
)
def test_a_directory_spelled_like_an_option_still_materializes(tier):
    """Reddens if a key's leading `-` reaches a command line, where `--version/` is a flag."""
    tree = {"--version/test_opt.py": "def test_opt():\n    assert True\n"}

    exit_code, output = run_tree_command(tree, ["ls", "--", "--version"], tier=tier)

    assert (exit_code, output.split()) == (0, ["test_opt.py"])


@pytest.mark.adversarial
def test_a_workspace_seeded_with_an_unsafe_tree_is_refused_at_construction():
    """Reddens if a seeded key the tools would refuse is admitted: every later write is then
    refused under that key's name, and the model has no tool that removes it."""
    with pytest.raises(UnsafeTreePath):
        Workspace(tree={**PASSING, "./legacy.py": "X = 1\n"})


@pytest.mark.adversarial
@pytest.mark.adversarial
def test_run_suite_refuses_a_named_tree_whose_content_has_no_bytes_as_a_value():
    """Reddens if content that cannot be encoded reaches the archive, where it raises out of the
    tool: `write_file`'s lint refuses it, and a tree named to `run_suite` skips that lint."""
    refused = serve_tool("run_suite", {"tree": {**PASSING, "a.py": "x\udc80"}})

    assert isinstance(refused, ToolRefused)


@pytest.mark.adversarial
def test_run_suite_over_the_workspace_refuses_an_unsafe_tree():
    """Reddens if the unnamed `run_suite` arm materializes a tree changed after the workspace
    checked it, as the dict it was built from can be. Only the direct API has that arm."""
    tree = dict(PASSING)
    ws = Workspace(tree=tree)
    tree["./legacy.py"] = "X = 1\n"

    with pytest.raises(ToolError):
        run_tool(ws, "run_suite", {})


def a_project_beside_a_secret(tmp_path: Path) -> tuple[Path, Path]:
    """`project/` holding one module, and `outside/secret.py` beside it, unreachable by name."""
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "mod.py").write_text("VALUE = 1\n")
    (outside / "secret.py").write_text('API_TOKEN = "s3cr3t"\n')
    return project, outside


@pytest.mark.adversarial
@pytest.mark.parametrize(
    "spelling",
    ["absolute", "dotdot", "symlink", "nul"],
)
def test_the_semantic_tools_refuse_a_path_outside_the_project(tmp_path, spelling):
    """Reddens if a semantic tool reads a file outside `project_root`: nothing is written, but its
    content lands in a recorded op result."""
    project, outside = a_project_beside_a_secret(tmp_path)
    (project / "linked.py").symlink_to(outside / "secret.py")
    path = {
        "absolute": str(outside / "secret.py"),
        "dotdot": "../outside/secret.py",
        "symlink": "linked.py",
        "nul": "mod\x00.py",
    }[spelling]
    root = str(project)

    for tool, args in [
        ("semantic_rename", {"path": path, "line": 1, "column": 0, "new_name": "X"}),
        ("semantic_references", {"path": path, "line": 1, "column": 0}),
        ("declared_arms", {"path": path, "alias": "API_TOKEN"}),
    ]:
        result = serve_tool(tool, args, project_root=root)
        assert isinstance(result, ToolRefused), (tool, result)
        assert "s3cr3t" not in str(result)


@pytest.mark.adversarial
@pytest.mark.parametrize("reached", ["symlinked-module", "stdlib"])
def test_a_rename_that_would_change_a_file_outside_the_project_is_refused(tmp_path, reached):
    """Reddens if a rename asked from inside the project returns a diff of a file outside it,
    putting its content in a recorded op result by a route the input path never names."""
    project, outside = a_project_beside_a_secret(tmp_path)
    (project / "secret.py").symlink_to(outside / "secret.py")
    (project / "uses.py").write_text(
        "import json\nfrom secret import API_TOKEN\n\njson.dumps(API_TOKEN)\n"
    )
    column = {"symlinked-module": 11, "stdlib": 5}[reached]
    root = str(project)

    result = serve_tool(
        "semantic_rename",
        {"path": "uses.py", "line": 4, "column": column, "new_name": "X"},
        project_root=root,
    )

    assert isinstance(result, ToolRefused), result
    assert "s3cr3t" not in str(result)


@pytest.mark.adversarial
def test_declared_arms_refuses_a_union_whose_arms_are_defined_outside_the_project(tmp_path):
    """Reddens if an alias imported through a symlink answers with the names an outside file
    declares. A refusal rather than fewer arms, since a union missing an arm is a wrong answer."""
    project, outside = a_project_beside_a_secret(tmp_path)
    (outside / "secret.py").write_text(
        "class Hunter2: ...\nclass Key: ...\ntype U = Hunter2 | Key\n"
    )
    (project / "secret.py").symlink_to(outside / "secret.py")
    (project / "uses.py").write_text("from secret import U\n\ntype V = U\n")
    root = str(project)

    result = serve_tool("declared_arms", {"path": "uses.py", "alias": "V"}, project_root=root)

    assert isinstance(result, ToolRefused), result
    assert "Hunter2" not in str(result)


def test_declared_arms_answers_with_the_interpreters_own_types(tmp_path):
    """The fence is the project and the interpreter that runs it: a builtin or standard-library
    arm is a fact about the language, and refusing it would refuse most unions."""
    (tmp_path / "u.py").write_text(
        "from pathlib import Path\nclass A: ...\ntype U = A | int | str | Path\n"
    )
    root = str(tmp_path)

    assert set(serve_tool("declared_arms", {"path": "u.py", "alias": "U"}, project_root=root)) == {
        "A",
        "int",
        "str",
        "Path",
    }


@pytest.mark.adversarial
def test_references_name_only_files_inside_the_project(tmp_path):
    project, outside = a_project_beside_a_secret(tmp_path)
    (project / "secret.py").symlink_to(outside / "secret.py")
    (project / "uses.py").write_text(
        "import json\nfrom secret import API_TOKEN\n\njson.dumps(API_TOKEN)\n"
    )
    root = str(project)

    found = [
        serve_tool(
            "semantic_references",
            {"path": "uses.py", "line": 4, "column": column},
            project_root=root,
        )
        for column in (5, 11)
    ]

    paths = {Path(ref.rsplit(":", 2)[0]).resolve() for refs in found for ref in refs}
    assert {path.is_relative_to(project.resolve()) for path in paths} == {True}
