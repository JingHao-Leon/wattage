# wattage-cli

[wattage](https://github.com/faizannraza/wattage) is a token-spend profiler
and CI cost-regression gate for AI agents. It reads OTLP GenAI traces and
Claude Code session files, prices out what your agent actually spent, and
fails a PR when a change makes the agent more expensive.

This package is a thin launcher for the Python CLI — it contains no analysis
logic of its own. It runs the `wattage` Python package via
[`uvx`](https://docs.astral.sh/uv/) (preferred) or
[`pipx`](https://pipx.pypa.io/), whichever is installed, pinned to this npm
package's own version so `wattage-cli@X.Y.Z` always runs engine X.Y.Z.

## Quickstart

```sh
npx wattage-cli demo
npx wattage-cli report --claude-code
```

## Overriding the pinned engine version

Set `WATTAGE_VERSION` to run a different version of the Python package
without waiting for a new npm release:

```sh
WATTAGE_VERSION=0.2.1 npx wattage-cli report --claude-code
```

## Links

- Source and docs: https://github.com/faizannraza/wattage
- Project site: https://faizannraza.github.io/wattage/
