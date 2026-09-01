#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

#include "fp8_gemm_tinym.cuh"

namespace {

using sm120_tinym::FP8GemmTinyM;
using sm120_tinym::FP8GemmTinyMMmaM16;

void check_inputs(const at::Tensor &a, const at::Tensor &b, double alpha) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(a.get_device() == b.get_device(),
                "A and B must be on the same CUDA device");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "A and B must both be rank-2 tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "A and B must be contiguous row-major tensors");
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                    b.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "A and B must have dtype torch.float8_e4m3fn");
    TORCH_CHECK(a.size(1) == b.size(1),
                "A and B must have the same K dimension");
    TORCH_CHECK(std::isfinite(alpha), "alpha must be finite");
}

at::Tensor fp8_gemm_cuda(const at::Tensor &a, const at::Tensor &b,
                         double alpha) {
    check_inputs(a, b, alpha);

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(0);
    TORCH_CHECK(
        (m == 1 || m == 9) &&
            ((k == 5120 && n == 16384) ||
             (k == 6144 && n == 5120) ||
             (k == 5120 && n == 8192)),
        "unsupported shape M=", m, ", K=", k, ", N=", n,
        "; this submission intentionally supports only the six official points");

    const c10::cuda::CUDAGuard device_guard(a.device());
    at::Tensor y = at::empty(
        {m, n}, a.options().dtype(at::ScalarType::BFloat16));

    const auto *x_ptr =
        reinterpret_cast<const __nv_fp8_e4m3 *>(a.data_ptr());
    const auto *w_ptr =
        reinterpret_cast<const __nv_fp8_e4m3 *>(b.data_ptr());
    auto *y_ptr = reinterpret_cast<__nv_bfloat16 *>(y.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());
    const float scale = static_cast<float>(alpha);

    if (m == 1 && k == 5120 && n == 16384) {
        FP8GemmTinyM<1, 16384, 5120, 256, 128, 3, 1>::run(
            x_ptr, w_ptr, y_ptr, scale, 1.0f, stream);
    } else if (m == 1 && k == 6144 && n == 5120) {
        FP8GemmTinyM<1, 5120, 6144, 128, 128, 6, 1>::run(
            x_ptr, w_ptr, y_ptr, scale, 1.0f, stream);
    } else if (m == 1 && k == 5120 && n == 8192) {
        FP8GemmTinyM<1, 8192, 5120, 128, 256, 3, 1>::run(
            x_ptr, w_ptr, y_ptr, scale, 1.0f, stream);
    } else if (m == 9 && k == 5120 && n == 16384) {
        FP8GemmTinyMMmaM16<9, 16384, 5120, 32, 128, 2>::run(
            x_ptr, w_ptr, y_ptr, scale, 1.0f, stream);
    } else if (m == 9 && k == 6144 && n == 5120) {
        FP8GemmTinyMMmaM16<9, 5120, 6144, 64, 128, 7>::run(
            x_ptr, w_ptr, y_ptr, scale, 1.0f, stream);
    } else {
        FP8GemmTinyMMmaM16<9, 8192, 5120, 64, 128, 3>::run(
            x_ptr, w_ptr, y_ptr, scale, 1.0f, stream);
    }

    return y;
}

}  // namespace

TORCH_LIBRARY(sm120_fp8_gemm, module) {
    module.def("run(Tensor a, Tensor b, float alpha) -> Tensor");
}

TORCH_LIBRARY_IMPL(sm120_fp8_gemm, CUDA, module) {
    module.impl("run", &fp8_gemm_cuda);
}
