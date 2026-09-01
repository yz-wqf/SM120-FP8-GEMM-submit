#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
    python_bin="$repo_dir/.venv/bin/python"
else
    python_bin="python3"
fi

cd "$repo_dir"
"$python_bin" scripts/test_correctness.py "$@"
