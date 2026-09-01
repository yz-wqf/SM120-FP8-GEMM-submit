#!/usr/bin/env python3
"""Correctness-only runner using the official harness's exact oracle and bar."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

# A file launched as scripts/test_correctness.py gets scripts/ as sys.path[0].
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidate_kernel
from fp8_gemm_task_harness import (
    MS,
    SHAPES,
    make_case,
    max_rel_err,
    reference,
    run_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the correctness test requires a CUDA GPU")

    candidate_kernel.build()
    all_ok = True
    print(
        f"{'point':<25}{'candidate max-rel':>19}"
        f"{'cuBLASLt max-rel':>19}{'limit':>13}{'result':>9}"
    )

    for k, n in SHAPES:
        for m in MS:
            aq, wq, sa, sw, alpha = make_case(
                m, k, n, args.seed + m + k + n
            )
            ref = reference(aq, wq, alpha)
            baseline = run_baseline(aq, wq, sa, sw)
            prepared = candidate_kernel.prepare_weights(wq)
            output = candidate_kernel.fp8_gemm(aq, prepared, alpha)

            structure_ok = (
                output.dtype == torch.bfloat16
                and tuple(output.shape) == (m, n)
                and torch.isfinite(output.float()).all().item()
            )
            candidate_rel = (
                max_rel_err(output, ref)
                if structure_ok
                else math.inf
            )
            baseline_rel = max_rel_err(baseline, ref)
            limit = 2.0 * baseline_rel + 1.0e-3
            ok = structure_ok and candidate_rel <= limit
            all_ok &= ok
            point = f"M={m}, K={k}, N={n}"
            print(
                f"{point:<25}{candidate_rel:>19.3e}"
                f"{baseline_rel:>19.3e}{limit:>13.3e}"
                f"{'PASS' if ok else 'FAIL':>9}"
            )

    print("correctness:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
