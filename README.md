# SM120 FP8 Tiny-M GEMM

Shape-specialized FP8 E4M3 GEMM kernels for NVIDIA Blackwell SM120, compared
against cuBLASLt.

The kernels compute:

<div align="center">
  <strong>Y[M,N] = (x_scale · w_scale) · X[M,K] · W[N,K]<sup>T</sup></strong>
</div>

<br>

Inputs are row-major FP8 E4M3, accumulation is FP32, and output is row-major
BF16. The implementation intentionally supports only the six shapes below.

> **Design overview:** Read the **[short design write-up](DESIGN.md)** for the
> architecture, tiling, thread maps, and measured trade-offs.

## Official task: one-command run

The setter-provided `fp8_gemm_task_harness.py` imports
`candidate_kernel.py`, checks all six points against its FP32 oracle, and
then prints the complete cold-L2 latency table against
`torch._scaled_mm`/cuBLASLt:

~~~text
./run.sh
~~~

The command JIT-builds the PyTorch CUDA extension and runs the official
correctness plus benchmark protocol. The equivalent individual commands are:

~~~text
./scripts/build.sh
./scripts/test.sh
./scripts/benchmark.sh
~~~

The scripts prefer `PYTHON=/path/to/python`, then `.venv/bin/python` when
present, then `python3`.
The candidate calls only the hand-written kernels in
`include/fp8_gemm_tinym.cuh`; it does not call a prebuilt GEMM
implementation.

## Fresh-environment setup

The tested Python environment is 3.12 with PyTorch 2.13.0+cu130. On
Debian/Ubuntu, install the host pieces needed by virtualenv and Triton's timer:

~~~text
sudo apt-get install python3-venv python3-dev build-essential
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
./run.sh
~~~

CUDA 13.x with an SM120-capable NVCC and NVIDIA driver must already be
available. If `Python.h` cannot be installed, uninstalling Triton makes the
unchanged official harness use its slower built-in CUDA-event fallback:

~~~text
.venv/bin/python -m pip uninstall -y triton
./run.sh
~~~

## Official harness results

Target-hardware run on an NVIDIA GeForce RTX 5090 (SM120, 170 SMs, 96 MiB L2)
with PyTorch 2.13.0+cu130. The unchanged official harness used
`triton.testing.do_bench`: 25 warmups, 100 repetitions, a cold-L2 protocol,
and the median latency.

| Point (M,K,N) | Final strategy (BM,BN,BK,stages) | Candidate | cuBLASLt | Speedup | Candidate MAX_REL | cuBLASLt MAX_REL | Result |
|---|---|---:|---:|---:|---:|---:|---|
| (1,5120,16384) | scalar (1,256,128,3), CWG=1 | 60.4 us | 69.7 us | 1.15x | 3.884e-3 | 3.884e-3 | PASS |
| (9,5120,16384) | m16 (16,32,128,2), 4 MMA warps | 62.2 us | 64.2 us | 1.03x | 3.890e-3 | 3.890e-3 | PASS |
| (1,6144,5120) | scalar (1,128,128,6), CWG=1 | 26.6 us | 27.9 us | 1.05x | 3.887e-3 | 3.887e-3 | PASS |
| (9,6144,5120) | generic m16 (16,64,128,7), 8 MMA warps | 26.6 us | 35.2 us | 1.32x | 3.875e-3 | 3.875e-3 | PASS |
| (1,5120,8192) | scalar BK256, 2xK128 W TMA (1,128,256,3), CWG=1 | 32.9 us | 52.5 us | 1.59x | 3.863e-3 | 3.863e-3 | PASS |
| (9,5120,8192) | m16 (16,64,128,3), 8 MMA warps | 33.8 us | 39.9 us | 1.18x | 3.888e-3 | 3.888e-3 | PASS |

Geometric-mean speedup against `torch._scaled_mm`/cuBLASLt: **1.208x**.

## Native CUDA cuBLASLt comparison

This earlier direct CUDA comparison is retained as an additional performance
table. It used the target NVIDIA GeForce RTX 5090, strict cold-L2 input
rotation, 20 warmups, and 300 timed launches. Speedup is cuBLASLt latency
divided by Tiny-M latency, so a value above one favors Tiny-M.

| Shape (M,N,K) | Kernel | BM | BN | BK | Consumer organization | NUM_STAGES | Tiny-M | cuBLASLt | Speedup | MAX_REL | Result |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| (1,16384,5120) | scalar | 1 | 256 | 128 | NUM_CONSUMER_WG=1 (4 warps) | 3 | 0.0568 ms | 0.0658 ms | 1.158x | 0.00689655 | PASS |
| (1,5120,6144) | scalar | 1 | 128 | 128 | NUM_CONSUMER_WG=1 (4 warps) | 6 | 0.0249 ms | 0.0262 ms | 1.053x | 0 | PASS |
| (1,8192,5120) | scalar BK256, 2xK128 W TMA | 1 | 128 | 256 | NUM_CONSUMER_WG=1 (4 warps) | 3 | 0.0306 ms | 0.0324 ms | 1.059x | 0 | PASS |
| (9,8192,5120) | m16 LDSM | 16 | 64 | 128 | BN/8=8 consumer warps + 1 producer | 3 | 0.0308 ms | 0.0331 ms | 1.073x | 0.00440529 | PASS |
| (9,5120,6144) | generic m16 LDSM | 16 | 64 | 128 | 8 consumer warps + 1 producer | 7 | 0.0238 ms | 0.0283 ms | 1.189x | 0.000244141 | PASS |
| (9,16384,5120) | m16 LDSM | 16 | 32 | 128 | BN/8=4 consumer warps + 1 producer | 2 | 0.0566 ms | 0.0616 ms | 1.088x | 0 | PASS |

All six retained kernels outperform their paired cuBLASLt measurement and pass
correctness:

- Three shapes are bitwise identical to cuBLASLt.
- (1,16384,5120) has <code>MAX_ABS=0.125</code> and
  <code>MAX_REL=0.00689655</code>.
- (9,8192,5120) has <code>MAX_ABS=0.125</code> and
  <code>MAX_REL=0.00440529</code>.
- (9,5120,6144) has <code>MAX_ABS=MAX_REL=0.000244141</code>.
- Every shape has zero mismatches at tolerance 0.02.

Here BM, BN, and BK are CTA-level tile dimensions. The M=1 kernels use one
four-warp consumer group. Every default M=9 kernel uses one producer warp plus
BN/8 m16n8 tensor-core consumer warps.

## Kernel families

### M=1 scalar path

The scalar path uses rank-1 TMA for X, rank-2 128-byte-swizzled TMA for W,
16-byte vector shared loads, FP32 scalar FMA, and coalesced BF16 stores.
BN=256 lets each consumer thread reuse an X vector across two output columns.
BK=256 is represented as two legal swizzled K128 W boxes.

### Generic M=9 m16 path

The generic M=9 path stages a padded M16 X tile. TMA zero-fills rows 9–15,
LDSM loads native tensor-core fragments, and each consumer warp computes one
M16-by-N8 tile with <code>mma.sync.aligned.m16n8k32</code>. Only rows 0–8 are
stored.

## Documentation and source

- **[Short design write-up: DESIGN.md](DESIGN.md)**
- [Tiny-M kernel implementation](include/fp8_gemm_tinym.cuh)
- [Official-harness adapter](candidate_kernel.py)
- [PyTorch CUDA binding](src/torch_candidate_extension.cu)
- [Setter-provided correctness and benchmark harness](fp8_gemm_task_harness.py)
- [Build, correctness, and benchmark scripts](scripts)

`DESIGN.md` summarizes the CTA/warp/thread mappings, TMA pipeline, numerical
validation, and the main design trade-offs within the requested two-page
limit.
