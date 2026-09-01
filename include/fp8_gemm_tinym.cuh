#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sm120_tinym {

// The extracted kernel throws this only from host-side launch plumbing.
// Device execution remains asynchronous; callers may synchronize separately.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char *expression, const char *file, int line)
        : std::runtime_error(format(code, expression, file, line)), code_(code) {}

    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;

    static std::string format(cudaError_t code, const char *expression,
                              const char *file, int line) {
        char message[512];
        std::snprintf(message, sizeof(message),
                      "CUDA error for %s at %s:%d: %s", expression, file, line,
                      cudaGetErrorString(code));
        return message;
    }
};

inline void cuda_check(cudaError_t code, const char *expression,
                       const char *file, int line) {
    if (code != cudaSuccess) throw CudaError(code, expression, file, line);
}

#define SM120_TINYM_CUDA_CHECK(call) \
    ::sm120_tinym::cuda_check((call), #call, __FILE__, __LINE__)

// A host-created CUtensorMap passed by value as a kernel argument.
struct TMADescriptor {
    alignas(64) uint64_t raw[16];

    __device__ __forceinline__ void load_1d(uint32_t tile_coord0,
                                            uint64_t smem_addr,
                                            uint64_t mbar_addr) const {
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global."
            "mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
            :
            : "l"(smem_addr), "l"((const uint64_t *)raw), "r"(tile_coord0),
              "r"((uint32_t)mbar_addr)
            : "memory");
    }

    __device__ __forceinline__ void load_2d(uint32_t tile_coord0,
                                            uint32_t tile_coord1,
                                            uint64_t smem_addr,
                                            uint64_t mbar_addr) const {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global."
            "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :
            : "l"(smem_addr), "l"((const uint64_t *)raw), "r"(tile_coord0),
              "r"(tile_coord1), "r"((uint32_t)mbar_addr)
            : "memory");
    }

};

inline TMADescriptor create_tma_desc_1d_raw(
    const void *gmem_ptr, CUtensorMapDataType dtype,
    uint32_t global_dim0, uint32_t box_dim0,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    TMADescriptor desc{};
    auto *map = reinterpret_cast<CUtensorMap *>(&desc.raw);
    uint64_t global_dims[1] = {global_dim0};
    uint64_t global_strides[1] = {0};
    uint32_t box_dims[1] = {box_dim0};
    uint32_t element_strides[1] = {1};

    CUresult result = cuTensorMapEncodeTiled(
        map, dtype, 1, const_cast<void *>(gmem_ptr), global_dims,
        global_strides, box_dims, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char *error_string = nullptr;
        cuGetErrorString(result, &error_string);
        throw std::runtime_error(
            std::string("cuTensorMapEncodeTiled failed: ") +
            (error_string ? error_string : "unknown driver error"));
    }
    return desc;
}

inline TMADescriptor create_tma_desc_2d_raw(
    const void *gmem_ptr, CUtensorMapDataType dtype, size_t elem_bytes,
    uint32_t global_dim0, uint32_t global_dim1,
    uint32_t box_dim0, uint32_t box_dim1,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
    TMADescriptor desc{};
    auto *map = reinterpret_cast<CUtensorMap *>(&desc.raw);
    uint64_t global_dims[2] = {global_dim0, global_dim1};
    uint64_t global_strides[1] = {global_dim0 * elem_bytes};
    uint32_t box_dims[2] = {box_dim0, box_dim1};
    uint32_t element_strides[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        map, dtype, 2, const_cast<void *>(gmem_ptr), global_dims,
        global_strides, box_dims, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char *error_string = nullptr;
        cuGetErrorString(result, &error_string);
        throw std::runtime_error(
            std::string("cuTensorMapEncodeTiled failed: ") +
            (error_string ? error_string : "unknown driver error"));
    }
    return desc;
}

__device__ __forceinline__ void mbarrier_init(uint64_t *barrier,
                                               uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "l"(barrier), "r"(count)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_inval(uint64_t *barrier) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                 :
                 : "l"(barrier)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t barrier_addr,
                                                    uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"((uint32_t)barrier_addr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t barrier_addr) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :
                 : "r"((uint32_t)barrier_addr)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t barrier_addr,
                                               uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "SM120_TINYM_WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra SM120_TINYM_WAIT_LOOP;\n"
        "}\n"
        :
        : "r"((uint32_t)barrier_addr), "r"(phase)
        : "memory");
}

__device__ __forceinline__ uint32_t smem_u32(const void *pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void ldsm_m8n8_x4_b16(
    const void *source, uint32_t (&fragment)[4]) {
    uint32_t address = smem_u32(source);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(fragment[0]), "=r"(fragment[1]),
          "=r"(fragment[2]), "=r"(fragment[3])
        : "r"(address)
        : "memory");
}

__device__ __forceinline__ void ldsm_m8n8_x2_b16(
    const void *source, uint32_t (&fragment)[2]) {
    uint32_t address = smem_u32(source);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 "
        "{%0, %1}, [%2];"
        : "=r"(fragment[0]), "=r"(fragment[1])
        : "r"(address)
        : "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

template <int SWIZZLE_BYTES, int ELEM_BYTES = 1>
__device__ __forceinline__ int swizzle_smem_offset(int row, int column,
                                                   int row_elements) {
    if constexpr (SWIZZLE_BYTES == 0) {
        return row * row_elements + column;
    } else {
        constexpr int kBanks = SWIZZLE_BYTES / 16;
        int column_bytes = column * ELEM_BYTES;
        int segment = column_bytes / SWIZZLE_BYTES;
        int within_segment = column_bytes % SWIZZLE_BYTES;
        int bank = within_segment >> 4;
        int offset = within_segment & 0xf;
        int swizzled_bank = bank ^ (row & (kBanks - 1));
        int swizzled_column =
            (segment * SWIZZLE_BYTES + swizzled_bank * 16 + offset) /
            ELEM_BYTES;
        return row * row_elements + swizzled_column;
    }
}

// A 128-byte TMA swizzle cannot encode a 256-element contiguous FP8 box.
// BK=256 is therefore staged as two independent [BN,128] swizzled subtiles.
// Keep the consumer address calculation here so the scalar FMA loop sees the
// same logical [BN,BK] coordinates for both BK=128 and BK=256 experiments.
template <int BK, int BN, int SWIZZLE_BYTES>
__device__ __forceinline__ int w_smem_offset(int row, int column) {
    if constexpr (BK == 256) {
        constexpr int kSubtileK = 128;
        int subtile = column / kSubtileK;
        int subtile_column = column % kSubtileK;
        return subtile * BN * kSubtileK +
               swizzle_smem_offset<SWIZZLE_BYTES, 1>(
                   row, subtile_column, kSubtileK);
    } else {
        return swizzle_smem_offset<SWIZZLE_BYTES, 1>(row, column, BK);
    }
}

__device__ __forceinline__ float2 fp8x2_to_f32x2(uint32_t packed) {
    __half2_raw half = __nv_cvt_fp8x2_to_halfraw2(
        (__nv_fp8x2_storage_t)(packed & 0xffffu), __NV_E4M3);
    return __half22float2(__half2(half));
}

// One aligned 16-byte shared-memory vector, widened from 16 e4m3 values.
__device__ __forceinline__ void load_fp8x16_f32(const __nv_fp8_e4m3 *source,
                                                float (&output)[16]) {
    uint4 raw = *reinterpret_cast<const uint4 *>(source);
    const uint32_t packed[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            float2 value = fp8x2_to_f32x2(packed[word] >> (16 * half));
            output[4 * word + 2 * half] = value.x;
            output[4 * word + 2 * half + 1] = value.y;
        }
    }
}

// Warp-level FP8 tensor-core operation used by the dedicated M=9 path.
// One warp computes a 16x8 output tile for K=32. Operand fragments use the
// PTX-prescribed m16n8k32 row/column register layout.
__device__ __forceinline__ void mma_m16n8k32_e4m3_f32(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// Dedicated M<=16 tensor-core kernel. Unlike the scalar Tiny-M path, this
// path stages a complete 16-row A tile. Runtime rows above M are supplied as
// TMA out-of-bounds zeros and are never stored.
template <int M, int N, int K, int BN, int BK, int NUM_STAGES>
struct FP8GemmTinyMMmaM16 {
    static constexpr int kProducerWarps = 1;
    static constexpr int kConsumerWarps = BN / 8;
    static constexpr int kTotalWarps = kProducerWarps + kConsumerWarps;
    static constexpr int kTotalThreads = kTotalWarps * 32;
    static constexpr int kTransactionBytes = (16 * BK + BN * BK);
    static constexpr int kSwizzleBytes = 128;

    static_assert(M > 1 && M <= 16, "m16 path requires 1 < M <= 16");
    static_assert(BN % 8 == 0, "BN must contain whole n8 MMA tiles");
    static_assert(BK % 32 == 0, "BK must contain whole k32 MMA tiles");
    static_assert(BK == 128,
                  "initial m16 experiment uses one 128B-swizzled K tile");
    static_assert(kTotalThreads <= 1024, "CTA exceeds CUDA thread limit");
    static_assert(NUM_STAGES >= 2, "at least two stages are required");

    struct SharedStorage {
        __nv_fp8_e4m3 w[NUM_STAGES][BN * BK];
        __nv_fp8_e4m3 x[NUM_STAGES][16 * BK];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int kSharedBytes = sizeof(SharedStorage);

    static void run(const __nv_fp8_e4m3 *__restrict__ x,
                    const __nv_fp8_e4m3 *__restrict__ w,
                    __nv_bfloat16 *__restrict__ y,
                    float x_scale, float w_scale,
                    cudaStream_t stream = nullptr);
};

template <int M, int N, int K, int BN, int BK, int NUM_STAGES>
__global__ void __launch_bounds__(
    FP8GemmTinyMMmaM16<M, N, K, BN, BK, NUM_STAGES>::kTotalThreads, 1)
fp8_gemm_tinym_mma_m16_kernel(
    float scale, __grid_constant__ const TMADescriptor tma_x,
    __grid_constant__ const TMADescriptor tma_w,
    __nv_bfloat16 *__restrict__ output) {
    using Params = FP8GemmTinyMMmaM16<M, N, K, BN, BK, NUM_STAGES>;
    using SharedStorage = typename Params::SharedStorage;

    extern __shared__ __align__(1024) char shared_raw[];
    auto &shared = *reinterpret_cast<SharedStorage *>(shared_raw);

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    constexpr int kTilesK = K / BK;

    if (thread_id == 0) {
        #pragma unroll
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            mbarrier_init(&shared.full_barrier[stage], 1);
            mbarrier_init(&shared.empty_barrier[stage],
                          Params::kConsumerWarps);
        }
    }
    __syncthreads();
    fence_proxy_async_shared();

    if (warp_id == 0) {
        if (lane_id == 0) {
            int stage = 0;
            int phase = 0;
            #pragma unroll 1
            for (int k_tile = 0; k_tile < kTilesK; ++k_tile) {
                if (k_tile >= NUM_STAGES) {
                    mbarrier_wait(
                        smem_u32(&shared.empty_barrier[stage]), phase ^ 1);
                }
                mbarrier_expect_tx(
                    smem_u32(&shared.full_barrier[stage]),
                    Params::kTransactionBytes);
                // The descriptor box is 16 rows high while global M is 9;
                // TMA zero-fills rows 9..15 for the m16 MMA operand.
                tma_x.load_2d(
                    k_tile * BK, 0, smem_u32(shared.x[stage]),
                    smem_u32(&shared.full_barrier[stage]));
                tma_w.load_2d(
                    k_tile * BK, blockIdx.x * BN,
                    smem_u32(shared.w[stage]),
                    smem_u32(&shared.full_barrier[stage]));
                if (++stage == NUM_STAGES) {
                    stage = 0;
                    phase ^= 1;
                }
            }
        }
    } else {
        const int consumer_warp = warp_id - 1;
        const int group = lane_id >> 2;
        const int thread_in_group = lane_id & 3;
        // M<=16 has exactly one MMA tile in M. A conventional 2D CTA raster
        // swizzle therefore degenerates to this direct N-tile mapping. Each
        // CTA streams a disjoint W slab, so permuting blockIdx.x cannot create
        // inter-CTA W reuse and would only disturb contiguous N ordering.
        const int warp_column_base =
            blockIdx.x * BN + consumer_warp * 8;
        float accumulators[4] = {};
        int stage = 0;
        int phase = 0;

        #pragma unroll 1
        for (int k_tile = 0; k_tile < kTilesK; ++k_tile) {
            __syncwarp();
            mbarrier_wait(smem_u32(&shared.full_barrier[stage]), phase);
            const __nv_fp8_e4m3 *shared_x = shared.x[stage];
            const __nv_fp8_e4m3 *shared_w = shared.w[stage];

            #pragma unroll
            for (int kk = 0; kk < BK; kk += 32) {
                uint32_t a[4];
                uint32_t b[2];
                // Reinterpret each pair of FP8 values as one b16 LDSM
                // element. The x4 matrix order matches the PTX A-fragment
                // register order: top-left, bottom-left, top-right,
                // bottom-right quadrants of the 16x32 FP8 tile.
                int matrix = lane_id >> 3;
                int matrix_row =
                    (lane_id & 7) + ((matrix & 1) ? 8 : 0);
                int matrix_k = kk + ((matrix >> 1) ? 16 : 0);
                ldsm_m8n8_x4_b16(
                    &shared_x[swizzle_smem_offset<128, 1>(
                        matrix_row, matrix_k, BK)],
                    a);

                // W[N,K] stores the MMA B columns as shared-memory rows.
                // Non-transposed LDSM therefore reproduces the PTX B-register
                // fragments directly; .trans would permute the packed FP8
                // values a second time.
                int b_matrix = (lane_id >> 3) & 1;
                int w_row = consumer_warp * 8 + (lane_id & 7);
                int w_k = kk + b_matrix * 16;
                ldsm_m8n8_x2_b16(
                    &shared_w[swizzle_smem_offset<128, 1>(
                        w_row, w_k, BK)],
                    b);
                mma_m16n8k32_e4m3_f32(accumulators, a, b);
            }

            __syncwarp();
            if (lane_id == 0) {
                mbarrier_arrive(smem_u32(&shared.empty_barrier[stage]));
            }
            __syncwarp();
            if (++stage == NUM_STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }

        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            int row = group + (element >= 2 ? 8 : 0);
            int column = warp_column_base +
                         thread_in_group * 2 + (element & 1);
            if (row < M) {
                output[row * N + column] =
                    __float2bfloat16(accumulators[element] * scale);
            }
        }
    }

    __syncthreads();
    if (thread_id == 0) {
        #pragma unroll
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            mbarrier_inval(&shared.full_barrier[stage]);
            mbarrier_inval(&shared.empty_barrier[stage]);
        }
    }
}

// Computes Y[M,N] = (x_scale * w_scale) * X[M,K] * W[N,K]^T.
// X and W contain e4m3 codes, accumulation is FP32, and Y is __nv_bfloat16.
template <int M, int N, int K, int BN, int BK, int NUM_STAGES,
          int NUM_CONSUMER_WG = 1>
struct FP8GemmTinyM {
    static constexpr int kWarpsPerWarpGroup = 4;
    static constexpr int kThreadsPerWarp = 32;
    static constexpr int kThreadsPerWarpGroup =
        kWarpsPerWarpGroup * kThreadsPerWarp;
    static constexpr int kTotalWarpGroups = NUM_CONSUMER_WG + 1;
    static constexpr int kTotalThreads =
        kTotalWarpGroups * kThreadsPerWarpGroup;
    static constexpr int kConsumerThreads = NUM_CONSUMER_WG * kThreadsPerWarpGroup;
    static constexpr int kTransactionBytes = (BK + BK * BN) * sizeof(__nv_fp8_e4m3);
    static constexpr int kVectorElements = 16;
    static constexpr int kNPerThread =
        (BN + kConsumerThreads - 1) / kConsumerThreads;
    static constexpr bool kEvenConsumerMapping =
        (BN % kConsumerThreads) == 0;
    static constexpr int kSwizzleBytes = 128;
    static constexpr int kSwizzleRowBlock = 8;
    static constexpr int kXStageRows = M == 1 ? 1 : kSwizzleRowBlock;

    static_assert(kNPerThread >= 1,
                  "each consumer thread must have a compile-time column slot");
    static_assert(BK % kVectorElements == 0,
                  "BK must be a multiple of 16 e4m3 elements");
    static_assert(BK * (int)sizeof(__nv_fp8_e4m3) >= 128,
                  "the 128-byte swizzle needs BK >= 128 for FP8");
    static_assert(BK == 128 || BK == 256,
                  "W staging supports BK128 or BK256 as two K128 boxes");
    static_assert(BN <= 256 && BK <= 256,
                  "TMA box dimensions are capped at 256");
    static_assert(BN % kSwizzleRowBlock == 0,
                  "BN must be a multiple of eight rows");
    static_assert(NUM_STAGES >= 2, "at least two pipeline stages are required");
    static_assert(NUM_CONSUMER_WG >= 1 && NUM_CONSUMER_WG <= 3,
                  "the measured search space uses one to three consumer groups");

    struct SharedStorage {
        // W is first so every W stage and each BK=256 128-wide subtile retain
        // at least 1024-byte alignment. M=1 X uses an unswizzled rank-1 TMA
        // load and therefore needs only its actual BK-byte row per stage.
        __nv_fp8_e4m3 w[NUM_STAGES][BK * BN];
        __nv_fp8_e4m3 x[NUM_STAGES][kXStageRows * BK];
        uint64_t full_barrier[NUM_STAGES];
        uint64_t empty_barrier[NUM_STAGES];
    };

    static constexpr int kSharedBytes = sizeof(SharedStorage);

    static void run(const __nv_fp8_e4m3 *__restrict__ x,
                    const __nv_fp8_e4m3 *__restrict__ w,
                    __nv_bfloat16 *__restrict__ y,
                    float x_scale, float w_scale,
                    cudaStream_t stream = nullptr);
};

template <int M, int N, int K, int BN, int BK, int NUM_STAGES,
          int NUM_CONSUMER_WG>
__global__ void __launch_bounds__(
    FP8GemmTinyM<M, N, K, BN, BK, NUM_STAGES, NUM_CONSUMER_WG>::kTotalThreads,
    1, 1)
fp8_gemm_tinym_kernel(float scale,
    __grid_constant__ const TMADescriptor tma_x,
    __grid_constant__ const TMADescriptor tma_w,
    __nv_bfloat16 *__restrict__ output) {
    using Params = FP8GemmTinyM<M, N, K, BN, BK, NUM_STAGES, NUM_CONSUMER_WG>;
    using SharedStorage = typename Params::SharedStorage;

    extern __shared__ __align__(1024) char shared_raw[];
    auto &shared = *reinterpret_cast<SharedStorage *>(shared_raw);

    constexpr int num_tiles_n = N / BN;
    constexpr int total_tiles = num_tiles_n;
    constexpr int k_end = K / BK;

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / Params::kThreadsPerWarp;
    const int lane_id = thread_id % Params::kThreadsPerWarp;
    const int warp_group_id = warp_id / Params::kWarpsPerWarpGroup;
    const int warp_in_group = warp_id % Params::kWarpsPerWarpGroup;
    const int block_count = gridDim.x;

    if (thread_id == 0) {
        #pragma unroll
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            mbarrier_init(&shared.full_barrier[stage], 1);
            mbarrier_init(&shared.empty_barrier[stage],
                          NUM_CONSUMER_WG * Params::kWarpsPerWarpGroup);
        }
    }
    __syncthreads();
    fence_proxy_async_shared();

    // Warp group zero is the producer. Only its first lane emits TMA loads;
    // the other 127 threads exist to preserve the warp-group role split.
    if (warp_group_id == 0) {
        if (warp_in_group == 0 && lane_id == 0) {
            int stage = 0;
            int phase = 0;
            int produced_k_tiles = 0;

            for (int tile_n = blockIdx.x;
                 tile_n < total_tiles;
                 tile_n += block_count) {
                for (int k_tile = 0; k_tile < k_end; ++k_tile) {
                    if (produced_k_tiles >= NUM_STAGES) {
                        mbarrier_wait(
                            smem_u32(&shared.empty_barrier[stage]), phase ^ 1);
                    }
                    mbarrier_expect_tx(
                        smem_u32(&shared.full_barrier[stage]),
                        Params::kTransactionBytes);
                    if constexpr (M == 1) {
                        tma_x.load_1d(
                            k_tile * BK, smem_u32(shared.x[stage]),
                            smem_u32(&shared.full_barrier[stage]));
                    } else {
                        tma_x.load_2d(
                            k_tile * BK, 0,
                            smem_u32(shared.x[stage]),
                            smem_u32(&shared.full_barrier[stage]));
                    }
                    if constexpr (BK == 256) {
                        // Tuple (BN,BK,stages)=(128,256,3) fits in 99,120
                        // bytes only with compact M=1 X staging. Stage W with
                        // two legal 128-byte-swizzled K128 TMA boxes; both
                        // arrivals contribute to one full-barrier byte sum.
                        tma_w.load_2d(
                            k_tile * BK, tile_n * BN,
                            smem_u32(shared.w[stage]),
                            smem_u32(&shared.full_barrier[stage]));
                        tma_w.load_2d(
                            k_tile * BK + 128, tile_n * BN,
                            smem_u32(shared.w[stage] + BN * 128),
                            smem_u32(&shared.full_barrier[stage]));
                    } else {
                        tma_w.load_2d(
                            k_tile * BK, tile_n * BN,
                            smem_u32(shared.w[stage]),
                            smem_u32(&shared.full_barrier[stage]));
                    }

                    if (++stage == NUM_STAGES) {
                        stage = 0;
                        phase ^= 1;
                    }
                    ++produced_k_tiles;
                }
            }
        }
    } else {
        const int consumer_thread = thread_id - Params::kThreadsPerWarpGroup;
        int stage = 0;
        int phase = 0;

        for (int tile_n = blockIdx.x;
             tile_n < total_tiles;
             tile_n += block_count) {

            float accumulators[Params::kNPerThread]{};

            for (int k_tile = 0; k_tile < k_end; ++k_tile) {
                __syncwarp();
                mbarrier_wait(smem_u32(&shared.full_barrier[stage]), phase);
                const __nv_fp8_e4m3 *shared_x = shared.x[stage];
                const __nv_fp8_e4m3 *shared_w = shared.w[stage];

                #pragma unroll
                for (int kk = 0; kk < BK; kk += Params::kVectorElements) {
                    // These X loads are shared-memory broadcasts, but the FP8
                    // widening is deliberately repeated by every consumer.
                    float x_values[Params::kVectorElements];
                    load_fp8x16_f32(
                        &shared_x[swizzle_smem_offset<
                            Params::kSwizzleBytes, 1>(0, kk, BK)],
                        x_values);

                    #pragma unroll
                    for (int column_index = 0;
                         column_index < Params::kNPerThread;
                         ++column_index) {
                        int column =
                            column_index * Params::kConsumerThreads +
                            consumer_thread;
                        if constexpr (Params::kEvenConsumerMapping) {
                            float w_values[Params::kVectorElements];
                            load_fp8x16_f32(
                                &shared_w[w_smem_offset<
                                    BK, BN, Params::kSwizzleBytes>(column, kk)],
                                w_values);

                            #pragma unroll
                            for (int element = 0;
                                 element < Params::kVectorElements; ++element) {
                                accumulators[column_index] = fmaf(
                                    x_values[element], w_values[element],
                                    accumulators[column_index]);
                            }
                        } else if (column < BN) {
                            // Oversubscribed consumer groups are measured by
                            // masking column slots beyond BN. Those warps still
                            // participate in all pipeline barriers.
                            float w_values[Params::kVectorElements];
                            load_fp8x16_f32(
                                &shared_w[w_smem_offset<
                                    BK, BN, Params::kSwizzleBytes>(column, kk)],
                                w_values);

                            #pragma unroll
                            for (int element = 0;
                                 element < Params::kVectorElements; ++element) {
                                accumulators[column_index] = fmaf(
                                    x_values[element], w_values[element],
                                    accumulators[column_index]);
                            }
                        }
                    }
                }

                __syncwarp();
                if (lane_id == 0) {
                    mbarrier_arrive(
                        smem_u32(&shared.empty_barrier[stage]));
                }
                __syncwarp();
                if (++stage == NUM_STAGES) {
                    stage = 0;
                    phase ^= 1;
                }
            }

            int tile_column_base = tile_n * BN;
            #pragma unroll
            for (int column_index = 0;
                 column_index < Params::kNPerThread; ++column_index) {
                int column = tile_column_base +
                             column_index * Params::kConsumerThreads +
                             consumer_thread;
                if constexpr (Params::kEvenConsumerMapping) {
                    float value = accumulators[column_index] * scale;
                    output[column] = __float2bfloat16(value);
                } else if (column - tile_column_base < BN) {
                    float value = accumulators[column_index] * scale;
                    output[column] = __float2bfloat16(value);
                }
            }
        }
    }

    __syncthreads();
    if (thread_id == 0) {
        #pragma unroll
        for (int stage = 0; stage < NUM_STAGES; ++stage) {
            mbarrier_inval(&shared.full_barrier[stage]);
            mbarrier_inval(&shared.empty_barrier[stage]);
        }
    }
}

template <int M, int N, int K, int BN, int BK, int NUM_STAGES,
          int NUM_CONSUMER_WG>
void FP8GemmTinyM<M, N, K, BN, BK, NUM_STAGES, NUM_CONSUMER_WG>::run(
    const __nv_fp8_e4m3 *__restrict__ x,
    const __nv_fp8_e4m3 *__restrict__ w, __nv_bfloat16 *__restrict__ y,
    float x_scale, float w_scale, cudaStream_t stream) {
    if (M < 1 || N < 1 || N % BN != 0 || K < 1 ||
        K % BK != 0) {
        throw std::invalid_argument(
            "Tiny-M requires 1 <= M, N % BN == 0, and K % BK == 0");
    }
    if (!x || !w || !y) {
        throw std::invalid_argument(
            "X, W, and Y must be non-null");
    }

    // The X descriptor's logical height is the runtime M. A BM-high TMA box
    // therefore zero-fills the overhanging rows when runtime M is smaller.
    TMADescriptor tma_x;
    if constexpr (M == 1) {
        tma_x = create_tma_desc_1d_raw(
            x, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, BK,
            CU_TENSOR_MAP_SWIZZLE_NONE);
    } else {
        tma_x = create_tma_desc_2d_raw(
            x, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, M, BK, 1,
            CU_TENSOR_MAP_SWIZZLE_128B);
    }
    constexpr int kWBoxK = BK == 256 ? 128 : BK;
    TMADescriptor tma_w = create_tma_desc_2d_raw(
        w, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N, kWBoxK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B);

    int device = 0;
    int multiprocessor_count = 0;
    SM120_TINYM_CUDA_CHECK(cudaGetDevice(&device));
    SM120_TINYM_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    // Launch at most one initial CTA per SM. If N/BN exceeds the SM count,
    // the grid-stride tile loop makes the first CTAs process the tail tiles.
    int block_count =
        std::min(multiprocessor_count, N / BN);

    auto kernel =
        fp8_gemm_tinym_kernel<M, N, K, BN, BK, NUM_STAGES, NUM_CONSUMER_WG>;
    SM120_TINYM_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    kernel<<<block_count, kTotalThreads, kSharedBytes, stream>>>(
        x_scale * w_scale, tma_x, tma_w, y);
    SM120_TINYM_CUDA_CHECK(cudaGetLastError());
}

template <int M, int N, int K, int BN, int BK, int NUM_STAGES>
void FP8GemmTinyMMmaM16<M, N, K, BN, BK, NUM_STAGES>::run(
    const __nv_fp8_e4m3 *__restrict__ x,
    const __nv_fp8_e4m3 *__restrict__ w,
    __nv_bfloat16 *__restrict__ y,
    float x_scale, float w_scale, cudaStream_t stream) {
    if (M < 2 || M > 16 || N < 1 || N % BN != 0 ||
        K < 1 || K % BK != 0) {
        throw std::invalid_argument(
            "m16 Tiny-M requires 2 <= M <= 16, N % BN == 0, and "
            "K % BK == 0");
    }
    if (!x || !w || !y) {
        throw std::invalid_argument("X, W, and Y must be non-null");
    }

    // A 16-row box deliberately overhangs the M=9 global tensor. TMA's
    // out-of-bounds handling supplies zeros for rows 9..15.
    TMADescriptor tma_x = create_tma_desc_2d_raw(
        x, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, M, BK, 16,
        CU_TENSOR_MAP_SWIZZLE_128B);
    TMADescriptor tma_w = create_tma_desc_2d_raw(
        w, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B);

    auto kernel =
        fp8_gemm_tinym_mma_m16_kernel<M, N, K, BN, BK, NUM_STAGES>;
    SM120_TINYM_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    kernel<<<N / BN, kTotalThreads, kSharedBytes, stream>>>(
        x_scale * w_scale, tma_x, tma_w, y);
    SM120_TINYM_CUDA_CHECK(cudaGetLastError());
}

template <int M, int N, int K, int BN, int BK, int NUM_STAGES>
inline cudaError_t sm120_tinym_mma_m16_run(
    const void *x, const void *w, void *y,
    float x_scale, float w_scale, cudaStream_t stream = nullptr) {
    try {
        FP8GemmTinyMMmaM16<M, N, K, BN, BK, NUM_STAGES>::run(
            static_cast<const __nv_fp8_e4m3 *>(x),
            static_cast<const __nv_fp8_e4m3 *>(w),
            static_cast<__nv_bfloat16 *>(y),
            x_scale, w_scale, stream);
        return cudaSuccess;
    } catch (const CudaError &error) {
        return error.code();
    }
}

}  // namespace sm120_tinym
