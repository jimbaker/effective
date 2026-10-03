# Pi's tool tests, translated

`test_pi_tools.py` translates the cases of Pi's tool tests that apply to `examples.coder`'s four
tools. Pi is MIT-licensed; its license is `LICENSE` beside this file and travels with the
translation.

| upstream | value |
|---|---|
| repository | https://github.com/earendil-works/pi |
| commit | `60e7e76bd` (2026-09-16) |
| files | `packages/coding-agent/test/tools.test.ts`, `packages/coding-agent/test/edit-tool-legacy-input.test.ts` |

Each row keeps Pi's `it(...)` title as its test id, so a row can be found upstream by searching
for the title. A row asserts what Pi's case asserts about behavior; wording differs where the coder
words its notices differently.

## Where the coder answers differently

| Pi's case | the coder's answer | why |
|---|---|---|
| `should handle command errors` | a nonzero exit is `Ran(exit_code=1)`, an observation | the model reads an exit code as data; Pi raises |
| `folds top-level oldText/newText into edits`, `parses edits from a JSON string` | `[bad arguments for edit]` | the coder calls the model with a strict function catalog, so the legacy spellings Pi repairs cannot arrive |
| the `edit tool CRLF handling` block, except LF preservation | an LF `old_text` does not match CRLF content, a BOM is content, and a CRLF/LF pair is two texts, so a duplicate across them is edited where Pi refuses | edits match bytes exactly. A strict xfail per case, until line-ending matching is implemented |

## Out of domain

| Pi's cases | why they do not apply |
|---|---|
| `should detect image MIME type from file magic (not extension)`, `should read BMP files from disk as PNG image attachments` | the project tree holds text only |
| `should include truncation details when truncated`, `should collapse large unchanged gaps in multi-edit diffs` | the coder's results carry no structured details or diff; the notice and the tree are the result |
| `should include EACCES for read-only files`, `should include the original error message for unknown edit access errors`, `should include ENOENT in diff preview for missing files`, `should include EACCES in diff preview for unreadable files` | the tree is a mapping with no permissions and no filesystem |
| `should throw error when cwd does not exist`, `should handle process spawn errors`, `should pass shellPath through to shell resolution`, `should send commands over stdin when shell resolution requires it`, `should resolve legacy WSL bash.exe to stdin command transport`, `should prepend command prefix when configured`, `should include output from both prefix and command`, `should work without command prefix` | `bash` runs `bash -c` in the pinned container from the project root |
| `should coalesce streaming updates for chatty output`, `should include full output path for truncated timeout and abort errors`, `should persist full output when truncation happens by line count only`, `executeBash should persist full output when truncation happens by line count only` | the coder keeps the tail within its caps and writes no full-output file; the container is discarded |
| `should expose local bash operations for extension reuse`, `should preserve executeBash sanitization when using local bash operations` | Pi's extension surface |
| the `grep tool` block, the `find tool` block, the `ls tool` block | the coder has four tools |
| the `tool cwd resolution` block | a path is a key in the project tree, relative to its root; there is no working directory |
| the `edit tool fuzzy matching` block, except the two cases that hold without it | fuzzy matching is not implemented |
| `appends legacy replacement to existing edits`, `passes through valid input unchanged`, `passes through non-object input unchanged`, `prepared args execute correctly`, `leaves edits alone when the string is not valid JSON` | Pi's argument repair, which the coder's strict catalog makes unnecessary |

`titles.txt` lists every title in the two files at the pinned commit, and
`test_every_upstream_case_is_translated_or_named` holds this page and the translation to it.
