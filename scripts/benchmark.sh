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
if "$python_bin" -c "import triton.testing" >/dev/null 2>&1; then
    echo "Timing backend: triton.testing.do_bench"
else
    echo "Timing backend: official 256-MiB-flush CUDA-event fallback"
fi
"$python_bin" fp8_gemm_task_harness.py --impl candidate_kernel "$@"
