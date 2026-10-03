# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""smol_agent — the whole agent, and nothing else.

A Python port of Thomas Schranz's 13-line Babashka (Clojure) agent, `smol.clj`
(https://x.com/__tosh/status/2085699009205743932). Stdlib only, so it is a fair weight against
that original (`bb` bundles its HTTP client and JSON the way Python bundles `urllib` and `json`).

Run it:  uv run examples/smol_agent.py https://host/v1/responses

**What this deliberately does NOT have**, because the omissions are the point:

- no durability — kill the process, or take one HTTP 500, and the run is gone;
- no permission tier — it runs whatever the model emits, as you, unsandboxed;
- no budget — the gauge is a readout, not a control.

It is short *because of* those three, and any comparison that forgets to say so is
selling something. What it does have is the state model the durable substrate agrees
with: `x` is an append-only list of items, model output and tool results alike, and the
agent loop is a fixpoint on "are there unanswered tool calls?".
"""

import json
import subprocess
import sys
import urllib.request
import uuid

URL, KEY, WINDOW = sys.argv[1], str(uuid.uuid4()), 10_500.0


def post(x: list[dict]) -> dict:
    tools = [{"type": "custom", "name": "sh"}]
    body = json.dumps({"model": "gpt-5.6-sol", "input": x, "tools": tools})
    req = urllib.request.Request(
        URL, body.encode(), {"content-type": "application/json", "session_id": KEY}
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def sh(call: dict) -> dict:
    # `stderr=STDOUT`, NOT `capture_output=True`. The latter is the obvious spelling and it
    # gives two separate pipes, so concatenating them reorders the output: `echo one; echo
    # two >&2; echo three` arrives as one/three/two. The model is then reading a false
    # account of what happened. Merging preserves the interleaving a human at a terminal
    # sees — which is the whole contract of a tool result.
    z = subprocess.run(
        ["/bin/sh", "-c", call["input"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "type": "custom_tool_call_output",
        "call_id": call["call_id"],
        "output": f"exit {z.returncode}\n{z.stdout}",
    }


x: list[dict] = []
while s := input("> ").strip():
    x.append({"role": "user", "content": s})
    while True:
        r = post(x)
        x += (o := r["output"])
        if not (c := [i for i in o if i["type"] == "custom_tool_call"]):
            gauge = r["usage"]["total_tokens"] / WINDOW
            print(o[-1]["content"][0]["text"], f"[{gauge:05.2f}%]", sep="\n")
            break
        x += [sh(i) for i in c]
