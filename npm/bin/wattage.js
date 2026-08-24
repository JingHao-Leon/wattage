#!/usr/bin/env node
"use strict";

// npx wattage-cli shim: wattage's actual implementation is the Python core
// (src/wattage/) — this just shells out to it via uvx (preferred, since it
// needs no separate install step) or pipx, whichever is available. No
// bundled/reimplemented logic here; the Python package is the single
// source of truth.
//
// The Python package is pinned to this npm package's own version, so
// `wattage-cli@X.Y.Z` always runs engine X.Y.Z rather than whatever
// happens to be newest on PyPI. Set WATTAGE_VERSION to override the pin.

const { spawnSync } = require("child_process");

const args = process.argv.slice(2);
const version =
  process.env.WATTAGE_VERSION || require("../package.json").version;
const spec = `wattage==${version}`;

function commandExists(command) {
  const probe = spawnSync(command, ["--version"], { stdio: "ignore" });
  return !probe.error;
}

function run(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { stdio: "inherit" });
  if (result.error) {
    console.error(`Failed to run ${command}: ${result.error.message}`);
    process.exit(1);
  }
  process.exit(result.status === null ? 1 : result.status);
}

if (commandExists("uvx")) {
  run("uvx", ["--from", spec, "wattage", ...args]);
} else if (commandExists("pipx")) {
  run("pipx", ["run", "--spec", spec, "wattage", ...args]);
} else {
  console.error(
    "wattage-cli needs either `uvx` (https://docs.astral.sh/uv/) or " +
      "`pipx` (https://pipx.pypa.io/) installed to run the Python core.\n" +
      "Install uv (curl -LsSf https://astral.sh/uv/install.sh | sh), then re-run:\n" +
      `  npx wattage-cli ${args.join(" ")}`
  );
  process.exit(2);
}
