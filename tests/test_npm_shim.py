"""Exercises the actual npm/bin/wattage.js shim as a real subprocess (not
just read for correctness) — the same scenarios manually verified during
development: a working `uvx`-like tool receives the pinned-version spec
(`--from wattage==<npm version> wattage`), forwards args, and propagates the
exit code (both a passing and a failing case); WATTAGE_VERSION overrides the
pin; and a clean error when neither uvx nor pipx is available.

`node` is invoked via its absolute path (not a bare "node" on PATH) so the
child's own PATH can be tightly controlled — on this machine node and the
*real* uvx both happen to live in the same Homebrew bin directory, so
naively adding "node's directory" to the test PATH would silently expose
the real uvx too and defeat the "neither available" test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SHIM = REPO_ROOT / "npm" / "bin" / "wattage.js"
VENV_WATTAGE = REPO_ROOT / ".venv" / "bin" / "wattage"
NPM_VERSION = json.loads((REPO_ROOT / "npm" / "package.json").read_text())["version"]

_NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(_NODE is None, reason="node not installed")

# System dirs needed for /usr/bin/env + bash to resolve inside the fake uvx
# shell script — not the real tool's PATH, just what a shebang needs.
_SYSTEM_BIN_DIRS = ["/usr/bin", "/bin"]


@pytest.fixture
def fake_uvx(tmp_path: Path) -> Path:
    """A stand-in `uvx` that asserts the shim pinned the Python package to
    the npm package's own version (`--from wattage==<version> wattage`),
    drops those leading args, and execs our real local CLI — lets us
    validate the shim's own pinning/arg-forwarding/exit-code logic without
    depending on wattage actually being published to PyPI."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    uvx_path = fake_bin / "uvx"
    uvx_path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--version" ]; then echo "fake-uvx 0.0.0"; exit 0; fi\n'
        f'if [ "$1" != "--from" ] || [ "$2" != "wattage=={NPM_VERSION}" ]'
        ' || [ "$3" != "wattage" ]; then\n'
        '  echo "fake-uvx: unexpected leading args: $*" >&2\n'
        "  exit 97\n"
        "fi\n"
        "shift 3\n"
        f'exec "{VENV_WATTAGE}" "$@"\n'
    )
    uvx_path.chmod(0o755)
    return fake_bin


def _run_shim(
    args: list[str],
    path_dirs: list[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert _NODE is not None
    path = ":".join(path_dirs)
    return subprocess.run(
        [_NODE, str(SHIM), *args],
        capture_output=True,
        text=True,
        env={"PATH": path, **(extra_env or {})},
        check=False,
    )


def test_forwards_args_and_propagates_a_passing_exit_code(fake_uvx: Path) -> None:
    result = _run_shim(
        ["score", str(REPO_ROOT / "examples" / "sample_trace.json")],
        [str(fake_uvx), *_SYSTEM_BIN_DIRS],
    )
    assert result.returncode == 0
    assert "A (100)" in result.stdout


def test_propagates_a_failing_exit_code(fake_uvx: Path) -> None:
    result = _run_shim(
        [
            "ci",
            str(REPO_ROOT / "examples" / "sample_trace.json"),
            "--fail-on",
            "garbage-not-a-clause",
        ],
        [str(fake_uvx), *_SYSTEM_BIN_DIRS],
    )
    assert result.returncode == 2


def test_wattage_version_env_var_overrides_the_pin(tmp_path: Path) -> None:
    """WATTAGE_VERSION is the escape hatch for running an engine version
    other than the npm package's own — the fake uvx here just echoes its
    args so we can see exactly which spec the shim asked for."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    uvx_path = fake_bin / "uvx"
    uvx_path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--version" ]; then echo "fake-uvx 0.0.0"; exit 0; fi\n'
        'echo "$@"\n'
    )
    uvx_path.chmod(0o755)
    result = _run_shim(
        ["--version"],
        [str(fake_bin), *_SYSTEM_BIN_DIRS],
        extra_env={"WATTAGE_VERSION": "9.9.9"},
    )
    assert result.returncode == 0
    assert "--from wattage==9.9.9 wattage --version" in result.stdout


def test_clean_error_when_neither_uvx_nor_pipx_available() -> None:
    result = _run_shim(["--version"], _SYSTEM_BIN_DIRS)
    assert result.returncode == 2
    assert "uvx" in result.stderr
    assert "pipx" in result.stderr
