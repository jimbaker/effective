"""Interpreters: what answers a domain op.

A handler interprets the workflow; an interpreter answers one `DomainOp` the handler hands it.

| module         | provides                                     | vendor                        |
|----------------|----------------------------------------------|-------------------------------|
| `jev`          | an interpreter for `Judge`, sync or asyncio  | `typesafe_sdk`, extra `judge` |
| `cli`          | an interpreter for `AskLLM`                  | `claude -p` or `codex exec`   |
| `openai`       | an interpreter for `AskLLM`, sync or asyncio | `openai`, imported where used |
| `aio`          | the loop an asyncio interpreter runs on      | none                          |
| `shell`        | a shell command a `CancelToken` can stop     | none                          |
| `tool_catalog` | the strict tool schemas an interpreter sends | none                          |

| a vendor is imported              | without it                                  |
|-----------------------------------|---------------------------------------------|
| at module load (`jev`)            | the import refuses, naming the `judge` extra |
| where it is used (`openai`)       | the module imports; the call needs the SDK  |
"""
