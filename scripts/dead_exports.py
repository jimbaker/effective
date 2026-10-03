"""A module's declared surface, bucketed by how many files outside it resolve to each name.

    uv run python scripts/dead_exports.py src/effective/graphview.py [roots...]

Roots default to `src tests`, because an export with one consumer under `src` may have twenty
under `tests`. A REPORT, never a gate: an exported return type, a protocol letter and a constant a
formal model quotes all read as dead here, and the judgement stays with the author.

`ast` beats ast-grep on this shape. One index over the tree answers every name; ast-grep would pay
a subprocess per name, which is 9.4s against 128ms at this population's size.
"""

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.whouses import Server, Site, _name_index

EXPORT_ROOTS = ("src", "tests")


def _declared_all(module: pathlib.Path) -> list[str]:
    """The names a module declares public, read off `__all__`."""
    for node in ast.parse(module.read_text()).body:
        match node:
            case ast.Assign(targets=[ast.Name(id="__all__")], value=ast.List(elts=elts)):
                return [
                    e.value
                    for e in elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
            case _:
                continue
    return []


def _defined_at(module: pathlib.Path, name: str) -> int | None:
    """Where the module defines `name`, so a resolved use can be checked against it."""
    for node in ast.parse(module.read_text()).body:
        match node:
            case ast.FunctionDef(name=n) | ast.AsyncFunctionDef(name=n) | ast.ClassDef(name=n) if (
                n == name
            ):
                return node.lineno
            case ast.Assign(targets=[ast.Name(id=n)]) | ast.AnnAssign(target=ast.Name(id=n)) if (
                n == name
            ):
                return node.lineno
            case ast.TypeAlias(name=ast.Name(id=n)) if n == name:
                # PEP 695 `type X = …`. Nine `__all__` entries are these, and without this arm all
                # nine were labelled "re-exported, defined elsewhere", false for every one, and it
                # took them out of the consumer analysis entirely. `src/` has 67 of them, so the
                # sweep was blind to a construct this repo showcases, which is the shape of the
                # PEP 634 hole this whole arc opened with.
                return node.lineno
            case _:
                continue
    return None


def _local_names(path: pathlib.Path, imported: str) -> set[str]:
    """What this file calls `imported`: itself, plus anything it aliased the import to.

    `from effective.parked import answer as answer_park` makes every use in that file an
    `answer_park`, so looking only under `answer` finds the import line and nothing else."""
    bound = {imported}
    for node in ast.walk(ast.parse(path.read_text())):
        match node:
            case ast.alias(name=name, asname=str() as alias) if (
                name.rpartition(".")[2] == imported
            ):
                bound.add(alias)
            case _:
                continue
    return bound


def _consumers(
    name: str,
    home: pathlib.Path,
    line: int,
    index: dict[str, list[Site]],
    server: Server,
) -> set[pathlib.Path]:
    """The files outside `home` that actually USE `home`'s `name`.

    Two rules, each of which was a wrong number before it was a rule. An `import` alias is not
    consumption: a package `__init__` that re-exports a name resolves to it and does nothing with
    it, so a file counts only on a site that is not an alias. And a file that renamed the import
    spells every use under the local name, so the candidate set has to include what this file calls
    it."""
    candidates: dict[pathlib.Path, list[Site]] = {}
    for site in index.get(name, []):
        if site.path.resolve() != home:
            candidates.setdefault(site.path, []).append(site)
    for path in list(candidates):
        for local in _local_names(path, name) - {name}:
            candidates[path].extend(site for site in index.get(local, []) if site.path == path)
    return {
        path
        for path, sites in candidates.items()
        if any(
            site.kind != "alias"
            and server.definition(site.path.resolve(), site.line - 1, site.column)
            == (str(home), line)
            for site in sites
        )
    }


def _report_exports(module: pathlib.Path, roots: list[str], server: Server) -> int:
    """A module's declared surface, bucketed by how many files outside it resolve to each name.

    **A report, not a gate.** Some of these are deliberate API: an exported return type, a
    protocol letter, a constant a formal model quotes, and the judgement stays with the author.
    A tool that proposed deletions while bounded by an enumerator would be the failure this file
    exists to document, and this one has already published a zero bucket that was wrong on nine
    names."""
    index = _name_index(roots)
    resolved = module.resolve()
    buckets: dict[int, list[str]] = {}
    for name in _declared_all(module):
        line = _defined_at(module, name)
        if line is None:
            buckets.setdefault(-1, []).append(name)
            continue
        buckets.setdefault(len(_consumers(name, resolved, line, index, server)), []).append(name)
    print(f"{module}  {sum(len(v) for v in buckets.values())} declared, over {' '.join(roots)}\n")
    for count in sorted(buckets):
        label = {
            -1: "re-exported, defined elsewhere",
            0: "NO consumer outside this file",
            1: "one consumer, generality nobody bought",
        }.get(count, f"{count} consumers")
        print(f"  {label}: {', '.join(sorted(buckets[count]))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(__doc__)
    module, roots = pathlib.Path(args[0]), list(args[1:]) or list(EXPORT_ROOTS)
    server = Server(pathlib.Path.cwd())
    try:
        return _report_exports(module, roots, server)
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())
