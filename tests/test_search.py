"""`effective.search`: UCT's selection, and MCTS on the recorder and under replay."""

import math
from collections.abc import Iterator, Mapping

import pytest

from effective.api import call_tool
from effective.budget import Grant
from effective.handlers.recording import RecordingHandler
from effective.handlers.replay import ReplayHandler
from effective.keys.grammar import parse
from effective.search import EXPLORATION, SearchTree, Stats, best_child, mcts, uct

SCORES = {"": 3.0, "a": 1.0, "b": 5.0, "aa": 0.0, "ab": 2.0, "ba": 9.0, "bb": 4.0}
"""A binary tree two levels deep, scored per node; `ba` is the best leaf."""


def expand(node: str):
    return (yield from call_tool("expand", {"node": node}, dict[str, str]))


def rollout(node: str):
    return (yield from call_tool("rollout", {"node": node}, float))


def path_of(key: str) -> str:
    return "".join(t.coordinates[0].atoms[0].text for t in parse(key).terms if t.tag == "node")


class Tree(Mapping[str, object]):
    """Answers by the `node:` frames on the key: a node below the leaves has no children."""

    def __init__(self, depth: int = 2) -> None:
        self.depth = depth
        self.expanded: list[str] = []

    def __getitem__(self, key: str) -> object:
        node = path_of(key)
        if key.endswith("tool:expand"):
            self.expanded.append(node)
            return {c: node + c for c in "ab"} if len(node) < self.depth else {}
        return SCORES[node]

    def __contains__(self, key: object) -> bool:
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def search(rounds: int = 4, depth: int = 3):
    return mcts("", expand, rollout, rounds=rounds, depth=depth)


def run(answers: Tree, rounds: int = 4) -> tuple[RecordingHandler, SearchTree[str]]:
    handler = RecordingHandler(answers)
    tree = handler.run(lambda: search(rounds))
    assert isinstance(tree, SearchTree)
    return handler, tree


UCT_CASES = {
    "an unvisited child goes first": ([Stats(2, 4.0), Stats(), Stats()], 2, 1),
    "no child visited, no parent visit": ([Stats(), Stats()], 0, 0),
    "the better mean, visits equal": ([Stats(2, 2.0), Stats(2, 6.0)], 4, 1),
    "exploration lifts the less visited": ([Stats(10, 10.0), Stats(1, 0.9)], 11, 1),
    "a tie goes to the earlier child": ([Stats(3, 3.0), Stats(3, 3.0)], 6, 0),
}


@pytest.mark.parametrize(("children", "parent", "chosen"), UCT_CASES.values(), ids=UCT_CASES)
def test_uct_chooses(children: list[Stats], parent: int, chosen: int) -> None:
    assert uct(children, parent) == chosen


def test_the_exploration_case_turns_on_the_exploration_term() -> None:
    children, parent, chosen = UCT_CASES["exploration lifts the less visited"]
    assert uct(children, parent, c=0.0) != chosen
    bonus = [s.mean + EXPLORATION * math.sqrt(math.log(parent) / s.visits) for s in children]
    assert bonus.index(max(bonus)) == chosen


def test_mcts_backpropagates_every_rollout_along_its_path() -> None:
    handler, tree = run(Tree())
    rolled = [path_of(e.key.stored()) for e in handler.trace if "tool:rollout" in e.key.stored()]
    for path, stats in tree.stats.items():
        under = [r for r in rolled if r.startswith("".join(path))]
        assert stats == Stats(len(under), sum(SCORES[r] for r in under)), path
    assert tree.stats[()].visits == len(rolled)


def test_mcts_follows_the_better_child() -> None:
    _, tree = run(Tree())
    assert best_child(tree) == max("ab", key=lambda c: SCORES[c])
    assert best_child(tree, ("b",)) == max(("a", "b"), key=lambda c: SCORES["b" + c])


def test_each_node_is_expanded_once() -> None:
    answers = Tree()
    run(answers, rounds=6)
    assert len(answers.expanded) == len(set(answers.expanded))


def test_a_terminal_root_is_rolled_out_every_round() -> None:
    answers = Tree(depth=0)
    _, tree = run(answers, rounds=3)
    assert answers.expanded == [""]
    assert tree.stats[()] == Stats(3, 3 * SCORES[""])
    assert best_child(tree) is None


def test_a_grantor_extends_the_rounds_until_it_stops() -> None:
    asked: list[int] = []

    def grantor(rounds: int):
        asked.append(rounds)
        yield from ()
        return Grant(add_rounds=2) if rounds < 4 else Grant(stop=True)

    handler = RecordingHandler(Tree())
    tree = handler.run(lambda: mcts("", expand, rollout, rounds=2, depth=3, grantor=grantor))
    assert isinstance(tree, SearchTree)
    assert asked == [2, 4]
    heads = [parse(e.key.stored()).terms[0] for e in handler.trace]
    assert {head.tag for head in heads} == {"search"}
    assert {int(head.coordinates[0].atoms[0].text) for head in heads} == set(range(4))


def test_replay_rebuilds_the_same_tree() -> None:
    handler, tree = run(Tree())
    assert ReplayHandler(handler.trace).run(search) == tree
