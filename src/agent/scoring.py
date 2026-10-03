"""Scorers — pure functions over a recorded answer, applied at *replay* time.

The cost thesis applied to evaluation: execution pays for the model once and records the
trajectory; **scoring is a pure function of the recorded answer**, so swapping or adding a
metric costs nothing (no model call). A `Scorer` maps `(answer, task) -> float` in [0, 1];
`agent.bench_telemetry.rescore` re-applies any scorer to already-recorded trajectories for
free.

This module is the official **HotpotQA answer** metrics (EM + token-F1), adapted from
`hotpot_evaluate_v1.py` (https://github.com/hotpotqa/hotpot, Apache-2.0) with its exact
normalization, so our quality number is comparable *in kind*
to the literature. (HotpotQA's supporting-facts / joint metrics need the agent to predict
supporting sentences, which this harness does not emit — out of scope, noted.)
"""

import re
import string
from collections import Counter
from collections.abc import Callable

from agent.tasks import Task

# A scorer reads a recorded answer + its task and returns a score in [0, 1].
Scorer = Callable[[str, Task], float]

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_YESNO = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    """The official HotpotQA/SQuAD normalization: lowercase, strip punctuation + articles,
    collapse whitespace."""
    return " ".join(_ARTICLES.sub(" ", s.lower().translate(_PUNCT)).split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 iff the normalized strings are identical, else 0.0 (official answer EM)."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Official HotpotQA token-level answer F1. yes/no/noanswer are scored by exact match
    (a wrong polarity earns 0), matching `hotpot_evaluate_v1.py`."""
    npred, ngold = normalize_answer(prediction), normalize_answer(gold)
    if npred in _YESNO or ngold in _YESNO:
        return float(npred == ngold)
    pred_tokens, gold_tokens = npred.split(), ngold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not pred_tokens or not gold_tokens:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def boolean_scorer(answer: str, task: Task) -> float:
    """The default scorer: the task's own `check` as 0.0/1.0 (preserves GSM8K pass-rate)."""
    return float(task.check(answer))
