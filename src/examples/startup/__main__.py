"""Run a startup workflow against its scripted world.

`uv run python -m examples.startup incident`, and likewise `launch` and `voice`.
"""

import argparse

from .engine import run
from .scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(prog="examples.startup")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    name = parser.parse_args().scenario
    scenario = SCENARIOS[name]
    world = scenario.world()
    ran = run(scenario.program, world, scenario.deliver)
    print("result:   ", ran.result)
    print("ops:      ", *ran.keys)
    print("parked on:", *ran.delivered or ("nothing",))
    print("calls:    ", dict(world.calls), "over", ran.attempts, "attempts")


if __name__ == "__main__":
    main()
