"""FP8 GEMM latency task -- official harness (correctness + benchmark).

Problem: out[M, N] = (A[M, K] @ B[N, K]^T) * alpha
  - A (activation) and B (weight): torch.float8_e4m3fn, contiguous row-major
  - alpha = a_scale * b_scale, two per-tensor fp32 scalars
  - accumulate in fp32, output bf16
Measurement points: M in {1, 9}  x  (K, N) in {(5120, 16384), (6144, 5120), (5120, 8192)}

Your submission is a module `candidate_kernel.py` next to this file, exposing:

    def fp8_gemm(a, b, alpha) -> torch.Tensor
        a:     [M, K] torch.float8_e4m3fn, contiguous
        b:     whatever `prepare_weights` returned (default: the raw [N, K] fp8 weight)
        alpha: [1] torch.float32
        returns [M, N] torch.bfloat16

    def prepare_weights(b) -> Any        # OPTIONAL
        One-time weight preprocessing (layout repack, etc.), like production does at
        model-load time. Runs once per shape, outside the timed region.

Any JIT compilation should happen at import / first call; the harness warms every
point up before timing. Do not call cuBLAS/cuBLASLt/CUTLASS prebuilt GEMMs inside
fp8_gemm -- that path is the baseline you are competing against.

Usage:
    python3 fp8_gemm_task_harness.py                        # check + bench candidate_kernel.py
    python3 fp8_gemm_task_harness.py --peak-gbps 1792       # % of peak uses this number
"""

import argparse
import importlib
import math

import torch

SHAPES = [(5120, 16384), (6144, 5120), (5120, 8192)]  # (K, N)
MS = [1, 9]
E4M3_MAX = 448.0


def quantize_per_tensor(x: torch.Tensor):
    scale = (x.abs().amax().float() / E4M3_MAX).clamp(min=1e-12)
    q = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(1)


def make_case(m: int, k: int, n: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32, generator=g) * 2.0
    w = torch.randn(n, k, device="cuda", dtype=torch.float32, generator=g) * 0.03
    # Exercise the full e4m3 dynamic range: plant values that land at +/-448
    # after per-tensor scaling, in both operands.
    a.view(-1)[::37] = 30.0
    a.view(-1)[17::74] = -30.0
    w.view(-1)[::1531] = 25.0
    w.view(-1)[733::3062] = -25.0
    aq, sa = quantize_per_tensor(a)
    wq, sw = quantize_per_tensor(w)
    alpha = (sa * sw).to(torch.float32).reshape(1)
    return aq, wq, sa, sw, alpha


def reference(aq, wq, alpha):
    # Exact fp32 matmul on the dequantized grid; the ground truth both sides chase.
    return (aq.float() @ wq.float().t()) * alpha


def run_baseline(aq, wq, sa, sw):
    return torch._scaled_mm(aq, wq.t(), scale_a=sa, scale_b=sw, out_dtype=torch.bfloat16)


def max_rel_err(x, ref):
    return ((x.float() - ref).abs() / ref.abs().clamp_min(1.0)).max().item()


def bench(fn) -> float:
    """Median latency in microseconds, cold-L2 protocol."""
    try:
        import triton.testing

        return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3
    except ImportError:
        # Fallback: manual L2 flush between iterations + CUDA events.
        flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        times = []
        for _ in range(50):
            flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1e3)
        times.sort()
        return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="candidate_kernel",
                        help="module exposing fp8_gemm (and optional prepare_weights)")
    parser.add_argument("--peak-gbps", type=float, default=1792.0,
                        help="theoretical memory bandwidth of your GPU, GB/s")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {dev.name}  (sm_{cap[0]}{cap[1]}), torch {torch.__version__}")
    if cap[0] != 12:
        print("WARNING: the task targets sm_120 (RTX 5090 / RTX PRO 6000); "
              f"you are on sm_{cap[0]}{cap[1]}.")

    impl = importlib.import_module(args.impl)
    prepare = getattr(impl, "prepare_weights", lambda b: b)

    print("\n=== correctness (vs fp32 reference on dequantized inputs; "
          "data spans the full e4m3 range incl. +/-448) ===")
    all_ok = True
    for k, n in SHAPES:
        for m in MS:
            aq, wq, sa, sw, alpha = make_case(m, k, n, args.seed + m + k + n)
            ref = reference(aq, wq, alpha)
            base = run_baseline(aq, wq, sa, sw)
            b_prep = prepare(wq)
            out = impl.fp8_gemm(aq, b_prep, alpha)
            ok = (out.dtype == torch.bfloat16 and tuple(out.shape) == (m, n)
                  and torch.isfinite(out.float()).all().item())
            cand_rel = max_rel_err(out, ref) if ok else float("inf")
            base_rel = max_rel_err(base, ref)
            # The bar is self-calibrating: you may differ from the baseline's
            # rounding, but not be meaningfully less accurate than it.
            ok = ok and cand_rel <= 2.0 * base_rel + 1e-3
            all_ok &= ok
            print(f"  M={m:<2d} K={k:<5d} N={n:<6d}  candidate_rel={cand_rel:.3e}  "
                  f"baseline_rel={base_rel:.3e}  {'PASS' if ok else 'FAIL'}")
    print("correctness:", "PASS" if all_ok else "FAIL")

    print("\n=== benchmark (cold L2, median of do_bench warmup=25 rep=100) ===")
    print(f"{'point':<24}{'baseline us':>12}{'candidate us':>14}{'speedup':>9}"
          f"{'cand GB/s':>11}{'% of peak':>10}")
    speedups = []
    for k, n in SHAPES:
        for m in MS:
            aq, wq, sa, sw, alpha = make_case(m, k, n, args.seed + m + k + n)
            b_prep = prepare(wq)
            impl.fp8_gemm(aq, b_prep, alpha)  # warm JIT before timing
            torch.cuda.synchronize()
            t_base = bench(lambda: run_baseline(aq, wq, sa, sw))
            t_cand = bench(lambda: impl.fp8_gemm(aq, b_prep, alpha))
            gbytes = (m * k + n * k + 2 * m * n) / 1e9
            gbps = gbytes / (t_cand * 1e-6)
            speedups.append(t_base / t_cand)
            print(f"  M={m:<2d} K={k:<5d} N={n:<6d} {t_base:>12.1f}{t_cand:>14.1f}"
                  f"{t_base / t_cand:>8.2f}x{gbps:>11.0f}{100 * gbps / args.peak_gbps:>9.1f}%")
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print(f"geomean speedup vs torch._scaled_mm (cuBLASLt): {geomean:.3f}x")


if __name__ == "__main__":
    main()
