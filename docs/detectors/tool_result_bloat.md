# tool_result_bloat

**Detects:** oversized tool results dumped into the context — a whole file read, a full API response, an uncapped shell command.

An oversized result doesn't just cost once: it rides along in every subsequent model call's context for the rest of the loop. This is one of the most common waste patterns in coding-agent sessions, and the fix is always at the tool boundary.

## How it works

Any tool result longer than `result_char_ceiling` (default 20,000 characters) is flagged; a worst offender at 4× the ceiling (configurable) makes the finding high severity. Findings aggregate per (task, tool): one oversized tool is reported once with its call count and worst offender, not once per call.

The dollar figure prices the **excess beyond the ceiling** at the task's input rate using the ~4-characters-per-token approximation, **once** — a deliberate under-estimate, since the real cost multiplies by the number of later turns carrying that text. It's still an estimate (basis `estimated`, shown with `~`), it's capped at the task's actual billed dollars so it can never exceed real spend, and it never counts toward the grade.

## Known limitation

The chars-per-token conversion is approximate, and without content capture Wattage can't tell how many later turns actually re-carried the result (nor whether it landed in a cached prefix). Both effects are documented rather than guessed at — which is exactly why this finding is `estimated`, not `measured`.

## Fix

Truncate, paginate, or summarize the tool's output at the tool boundary before it enters the context: head/tail a file read, cap search results, strip boilerplate from API responses.

## Quality risk: low

Trimming tool output is usually safe, but an aggressive truncation can drop the one line the agent needed — hence "low" rather than "none".
