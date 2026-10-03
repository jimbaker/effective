// The Node facade for ELK layout — one persistent process, JSON per line.
//
// Deliberately dumb: it knows nothing about Effective. Python decides what the
// graph MEANS (effective/graphlayout/prepare.py); this decides only where the
// boxes go. Requests and responses are one JSON object per line so a single
// process can serve many layouts without paying node's ~60 ms startup each time.
//
// Protocol
//   in : {"id": "<request id>", "graph": <ELK JSON>, "options": {…}?}
//   out: {"id": …, "ok": true,  "result": <ELK JSON + geometry>, "engine_version": "0.12.0"}
//        {"id": …, "ok": false, "error": "…", "stack": "…"}
//
// A malformed line answers with ok:false and keeps the process alive; the Python
// client (effective.graphlayout.elkjs) owns timeouts and restarts.

import readline from "node:readline";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ELK = require("elkjs/lib/elk.bundled.js");
const { version } = require("elkjs/package.json");

const elk = new ELK();

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

function reply(payload) {
  process.stdout.write(JSON.stringify(payload) + "\n");
}

for await (const line of input) {
  if (!line.trim()) {
    continue;
  }

  let id = null;

  try {
    const request = JSON.parse(line);
    id = request.id ?? null;
    const result = await elk.layout(request.graph, {
      layoutOptions: request.options ?? {},
    });
    reply({ id, ok: true, result, engine_version: version });
  } catch (error) {
    reply({
      id,
      ok: false,
      error: String(error),
      stack: error?.stack ?? null,
      engine_version: version,
    });
  }
}
