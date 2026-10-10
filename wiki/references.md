# References

Work this project builds on, follows or ships. Source code names these by a bare proper noun
(RLM, GEPA, ReAct); this page resolves the noun. Common ideas with an academic origin (algebraic
effects, replay-based durable execution, beam search, MCTS, Pareto frontiers, value of
information) are not listed.

## Vendored, translated or adapted

Each carries its license in the tree, and the file named here is the record.

| what | used in | source | license | record |
|---|---|---|---|---|
| Absurd, the Postgres durable-execution schema and its migration | `infra/absurd/`, `effective.engines.absurd`, `effective.absurd_worker` | Armin Ronacher and Earendil, *Absurd*, https://github.com/earendil-works/absurd, tag 0.5.0 | Apache-2.0 | [`infra/absurd/PIN.txt`](../infra/absurd/PIN.txt) |
| tdom, the t-string HTML templating engine, with one patch | `infra/tdom/`, `effective.graphlayout.svg` | Dave Peck, Ian Wilson, Andrea Giammarchi and Paul Everitt, *tdom*, https://github.com/t-strings/tdom, 0.1.17 | MIT | [`infra/tdom/PIN.txt`](../infra/tdom/PIN.txt) |
| Pi's coding-agent tool tests, translated from TypeScript | `tests/conformance/pi/`, `examples.coder` | Mario Zechner, *pi*, https://github.com/earendil-works/pi | MIT | [`tests/conformance/pi/SOURCE.md`](../tests/conformance/pi/SOURCE.md) |
| The GSM8K sample | [`src/agent/data/gsm8k_sample.jsonl`](../src/agent/data/gsm8k_sample.jsonl), `agent.gsm8k` | Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021, arXiv:2110.14168, https://github.com/openai/grade-school-math | MIT | [`src/agent/data/README.md`](../src/agent/data/README.md) |
| The HotpotQA answer metrics (EM and token F1), adapted | `agent.scoring` | Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", EMNLP 2018, arXiv:1809.09600; `hotpot_evaluate_v1.py` in https://github.com/hotpotqa/hotpot | Apache-2.0 | the `agent.scoring` docstring |

## Methods and systems followed

| what | used in | cite |
|---|---|---|
| Recursive Language Models (RLM) | `effective.code`, `effective.combinators` (`run_code`, `recurse`, `route`, `descend`), `agent.contrastbench` | Alex L. Zhang, Tim Kraska and Omar Khattab, "Recursive Language Models", 2025, arXiv:2512.24601; DSPy's `dspy.RLM`, https://dspy.ai/api/modules/RLM/ |
| GEPA, reflective prompt evolution with Pareto selection | `effective.improve` | Lakshya A Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", ICLR 2026, arXiv:2507.19457; "actionable side information" is the library's term, https://github.com/gepa-ai/gepa |
| ReAct | `effective.react` (`run_agent`), the `agent` benches | Shunyu Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023, arXiv:2210.03629 |
| Locally-in-distribution harnesses, the `~_H` equivalence | `effective.lineage` | Alex L. Zhang, "Language model harnesses are compositional generalizers", 2026, https://alexzhang13.github.io/blog/2026/harness/ |
| SkillsBench and its Harbor harness | `agent.skillsbench` | Xiangyi Li et al., "SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks", 2026, arXiv:2602.12670, https://github.com/benchflow-ai/skillsbench |
| The Agent Skills format (`SKILL.md`) | `effective.skills` | Anthropic, *Agent Skills*, https://agentskills.io/specification |
| `smol.clj`, a 13-line Babashka agent, ported | [`examples/smol_agent.py`](../examples/smol_agent.py), `effective.smol` | Thomas Schranz (@__tosh), https://x.com/__tosh/status/2085699009205743932 |
| TypeSafe's typed-judgment models (Jev) and their Python SDK | `effective.interpreters.jev`, the `judge` extra, the judge examples | TypeSafe AI, https://typesafe.ai; `typesafe-sdk`, https://github.com/typesafe-ai/typesafe-sdk-python (MIT) |
| Pydantic Monty, the sandboxed Python interpreter | `effective.monty` | Pydantic, *Monty*, https://github.com/pydantic/monty |
| OpenTelemetry GenAI semantic conventions | `effective.telemetry`, `effective.cost` | https://github.com/open-telemetry/semantic-conventions-genai |
| Cordis and DeepSeek Harness, as comparison | [`docs/repertoire-axis.md`](../docs/repertoire-axis.md) | Yifan Shi, Wei Zhang and Tianyi Cui, "A Programming Paradigm for Spatiotemporal Composability", 2026, arXiv:2608.25512; DeepSeek, *DeepSeek Harness*, https://github.com/deepseek-ai/deepseek-harness |

## Background

Orientation for a reader new to the field, rather than the source of any one design.

| what | where |
|---|---|
| CMU 11-768, *AI Agents*: agent architectures, tools, search and deep research | https://www.cmu-agents.com |
| Stanford CS329Z, *Engineering AI Agents*: from tool use and scaffolds through optimization, evaluation, guardrails and coding agents | https://cs329z.stanford.edu |
| PEP 750, template strings, the language feature the data axis is built on | https://peps.python.org/pep-0750/ |

Upstream: [index](index.md).
