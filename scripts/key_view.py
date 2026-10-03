"""Every registered key shape, sorted by template: the minter population in one normal form.

A report, never a gate. The sort is the whole instrument. Rendering each variant with its holes
anonymous discards what differs incidentally (the function name, the file, the parameter spelling)
and keeps what differs essentially, so a namespace's siblings land on adjacent lines and a family
is visible without anything scoring similarity.

Reads `build/key-registry.json`; run `just key-registry` if it is stale.
"""

import sys

from effective.keys.registry import DEFAULT_PATH, KeyMap


def rows(keymap: KeyMap) -> list[tuple[str, tuple[str, ...]]]:
    """`(template, every producing site)` for each registered variant, sorted by template.

    Every site, because a variant records one `site` and arbitrarily many `sites`, and printing
    the first would hide the duplication this view exists to find: 21 of 103 producers sat behind
    82 variant rows when it did.
    """
    return sorted(
        (shape.template, tuple(shape.sites or (shape.site,)))
        for shapes in keymap.variants.values()
        for shape in shapes
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    keymap = KeyMap.load(args[0] if args else DEFAULT_PATH)
    table = rows(keymap)
    width = max((len(template) for template, _ in table), default=0)
    for template, sites in table:
        print(f"{template:<{width}}  {sites[0]}")
        for extra in sites[1:]:
            print(f"{'':<{width}}  {extra}")
    shared = [(template, sites) for template, sites in table if len(sites) > 1]
    total = sum(len(sites) for _t, sites in table)
    print(f"\n{len(table)} variants over {len(keymap.variants)} namespaces, {total} producers")
    print(f"{len(shared)} templates are minted at more than one site:")
    for template, sites in shared:
        print(f"  {template:<{width}}  {len(sites)} sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
