// cutlass_kernels.cu — thor-fp4's own sm_110a kernels, on CUTLASS 4.4.1
// (BSD-3; needs the two-line sm_110 enablement in patches/).
//
// What is here and why:
//   * NVFP4 x NVFP4 (kind::mxf4nvf4, UE4M3 scale per 16): the same GEMM
//     cuBLASLt runs, from an open-source kernel we can tune and inspect.
//   * MXFP8 x MXFP4 (kind::mxf8f6f4, UE8M0 scale per 32): FP8 activations x
//     FP4 weights — a mix cuBLASLt 13.4.1 refuses on sm_110 (heuristic
//     status 7/15 for every combination; measured) but the tensor cores take
//     through CUTLASS.
//   * MXFP4 x MXFP4: likewise refused by cuBLASLt on sm_110; this proves (or
//     disproves) the hardware path exists for pure MXFP4 too.
//
// All variants: A = activations (M,K) row-major, B = weights (N,K) row-major
// (CUTLASS sees (K,N) column-major), D = (M,N) row-major BF16. Block-scale
// buffers use the same tcgen05 128x4-tile layout as the core extension, so
// thor_fp4.quantize_nvfp4 / quantize_mx outputs feed these directly.
//
// Alignment note: e2m1 operands use alignment 128 (= 64 B) here. The lab's
// original nvfp4 harness used alignment 32 and its output disagreed with
// cuBLASLt at relRMS ~1.4; the mixed-input kernel that validated used 128
// for its e2m1 operand. Variant "nvfp4_..._a32" keeps alignment 32 to
// document the difference; everything else uses 128.
//
// C ABI:
//   tfc4_n_variants()             number of registered variants
//   tfc4_variant_name(i)          static name
//   tfc4_variant_index(name)      index or -1
//   tfc4_gemm(variant, A, SA, B, SB, D, M, N, K, alpha, stream)
//     D (M,N) bf16 row-major = alpha * (A_dequant @ B_dequant^T);
//     K must be a multiple of 64. Per (variant, M, N, K) the kernel params
//     and workspace are resolved once and cached, so repeat calls —
//     including under CUDA graph capture — only launch.
//   tfc4_cutlass_version()        CUTLASS_MAJOR*10000+MINOR*100+PATCH
//
// Build: nvcc -arch=sm_110a -O2 -std=c++17 --expt-relaxed-constexpr
//        -Xcompiler -fPIC -shared -I$CUTLASS/include -I$CUTLASS/tools/util/include
//        -o build/libthorfp4_cutlass.so csrc/cutlass_kernels.cu -lcublasLt

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/version.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

template <class Tile, class Cluster, class ElemA, int AlignA,
          class ElemB, int AlignB>
struct VT {
    using ElementA = ElemA;
    using ElementB = ElemB;
    using ElementD = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementAcc = float;
    using ArchTag = cutlass::arch::Sm100;   // compiled for sm_110a; see patches/
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OpClass, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        ElementC, cutlass::layout::RowMajor, 8,
        ElementD, cutlass::layout::RowMajor, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        ElementA, cutlass::layout::RowMajor, AlignA,
        ElementB, cutlass::layout::ColumnMajor, AlignB,
        ElementAcc, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// resolve+cache one (variant, M, N, K); subsequent calls only update pointers
// and launch
template <class V>
int ensure(int64_t M, int64_t N, int64_t K, const void* A, const void* SA,
           const void* B, const void* SB, void* D, float alpha,
           cudaStream_t stream) {
    using Gemm = typename V::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    static std::map<std::tuple<int64_t, int64_t, int64_t>,
                    std::pair<std::shared_ptr<Gemm>, std::shared_ptr<void>>> cache;
    auto key = std::make_tuple(M, N, K);
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto gemm = std::make_shared<Gemm>();
        StrideA stA = cutlass::make_cute_packed_stride(StrideA{}, {int(M), int(K), 1});
        StrideB stB = cutlass::make_cute_packed_stride(StrideB{}, {int(N), int(K), 1});
        StrideC stC = cutlass::make_cute_packed_stride(StrideC{}, {int(M), int(N), 1});
        StrideD stD = cutlass::make_cute_packed_stride(StrideD{}, {int(M), int(N), 1});
        auto lSFA = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFA(cute::make_shape(int(M), int(N), int(K), 1));
        auto lSFB = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFB(cute::make_shape(int(M), int(N), int(K), 1));
        // dummy 64B-aligned pointers for the one-time validate/size pass
        const void* dmy = (const void*)0x10000ull;
        typename Gemm::Arguments probe{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {int(M), int(N), int(K), 1},
            {reinterpret_cast<const typename V::ElementA::DataType*>(dmy), stA,
             reinterpret_cast<const typename V::ElementB::DataType*>(dmy), stB,
             reinterpret_cast<const typename V::ElementA::ScaleFactorType*>(dmy), lSFA,
             reinterpret_cast<const typename V::ElementB::ScaleFactorType*>(dmy), lSFB},
            {{1.0f, 0.0f}, nullptr, stC,
             reinterpret_cast<cutlass::bfloat16_t*>(const_cast<void*>(dmy)), stD}};
        if (gemm->can_implement(probe) != cutlass::Status::kSuccess)
            return -20;
        size_t ws_bytes = Gemm::get_workspace_size(probe);
        void* ws = nullptr;
        if (ws_bytes) {
            if (cudaMalloc(&ws, ws_bytes) != cudaSuccess)
                return -21;
        }
        auto wsg = std::shared_ptr<void>(ws, [](void* p) { if (p) cudaFree(p); });
        if (gemm->initialize(probe, ws) != cutlass::Status::kSuccess)
            return -22;
        it = cache.emplace(key, std::make_pair(gemm, wsg)).first;
    }
    // per-call arguments (real pointers), then a lightweight params update
    StrideA stA = cutlass::make_cute_packed_stride(StrideA{}, {int(M), int(K), 1});
    StrideB stB = cutlass::make_cute_packed_stride(StrideB{}, {int(N), int(K), 1});
    StrideC stC = cutlass::make_cute_packed_stride(StrideC{}, {int(M), int(N), 1});
    StrideD stD = cutlass::make_cute_packed_stride(StrideD{}, {int(M), int(N), 1});
    auto lSFA = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
        tile_atom_to_shape_SFA(cute::make_shape(int(M), int(N), int(K), 1));
    auto lSFB = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
        tile_atom_to_shape_SFB(cute::make_shape(int(M), int(N), int(K), 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {int(M), int(N), int(K), 1},
        {reinterpret_cast<const typename V::ElementA::DataType*>(A), stA,
         reinterpret_cast<const typename V::ElementB::DataType*>(B), stB,
         reinterpret_cast<const typename V::ElementA::ScaleFactorType*>(SA), lSFA,
         reinterpret_cast<const typename V::ElementB::ScaleFactorType*>(SB), lSFB},
        {{alpha, 0.0f}, nullptr, stC,
         reinterpret_cast<cutlass::bfloat16_t*>(D), stD}};
    if (it->second.first->update(args) != cutlass::Status::kSuccess)
        return -23;
    if (it->second.first->run(stream) != cutlass::Status::kSuccess)
        return -24;
    return 0;
}

using NvF4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using MxF4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using MxF8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;

struct Variant {
    const char* name;
    int (*fn)(int64_t, int64_t, int64_t, const void*, const void*,
              const void*, const void*, void*, float, cudaStream_t);
};

// the registry: tile shape x cluster x operand formats. e2m1 uses alignment
// 128 (= 64 B) except the documented alignment-32 nvfp4 variant.
const Variant kVariants[] = {
    {"nvfp4_128x128x256_c1",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_1, _1, _1>, NvF4, 128, NvF4, 128>>},
    {"nvfp4_128x128x256_c1_a32",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_1, _1, _1>, NvF4, 32, NvF4, 32>>},
    {"nvfp4_128x128x256_c2x1",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_2, _1, _1>, NvF4, 128, NvF4, 128>>},
    {"nvfp4_128x256x256_c2x1",
     &ensure<VT<Shape<_128, _256, _256>, Shape<_2, _1, _1>, NvF4, 128, NvF4, 128>>},
    {"nvfp4_128x128x128_c1",
     &ensure<VT<Shape<_128, _128, _128>, Shape<_1, _1, _1>, NvF4, 128, NvF4, 128>>},
    {"nvfp4_128x128x64_c1",
     &ensure<VT<Shape<_128, _128, _64>, Shape<_1, _1, _1>, NvF4, 128, NvF4, 128>>},
    {"mxfp8xfp4_128x128x256_c1",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_1, _1, _1>, MxF8, 32, MxF4, 128>>},
    {"mxfp8xfp4_128x128x256_c2x1",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_2, _1, _1>, MxF8, 32, MxF4, 128>>},
    {"mxfp8xfp4_128x256x256_c2x1",
     &ensure<VT<Shape<_128, _256, _256>, Shape<_2, _1, _1>, MxF8, 32, MxF4, 128>>},
    // K-tile 128: halves the small-K fixed cost; K%128 still required.
    // N-tile 256 swap: better N-tiling for very wide layers.
    {"mxfp8xfp4_128x128x128_c1",
     &ensure<VT<Shape<_128, _128, _128>, Shape<_1, _1, _1>, MxF8, 32, MxF4, 128>>},
    {"mxfp8xfp4_128x128x128_c2x1",
     &ensure<VT<Shape<_128, _128, _128>, Shape<_2, _1, _1>, MxF8, 32, MxF4, 128>>},
    {"mxfp8xfp4_128x256x128_c2x1",
     &ensure<VT<Shape<_128, _256, _128>, Shape<_2, _1, _1>, MxF8, 32, MxF4, 128>>},
    {"mxfp8xfp4_256x128x256_c2x1",
     &ensure<VT<Shape<_256, _128, _256>, Shape<_2, _1, _1>, MxF8, 32, MxF4, 128>>},
    {"mxfp4_128x128x256_c1",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_1, _1, _1>, MxF4, 128, MxF4, 128>>},
    {"mxfp4_128x128x256_c2x1",
     &ensure<VT<Shape<_128, _128, _256>, Shape<_2, _1, _1>, MxF4, 128, MxF4, 128>>},
};
constexpr int kNVariants = int(sizeof(kVariants) / sizeof(kVariants[0]));

}  // namespace

extern "C" {

int tfc4_n_variants() { return kNVariants; }

const char* tfc4_variant_name(int i) {
    return (i >= 0 && i < kNVariants) ? kVariants[i].name : nullptr;
}

int tfc4_variant_index(const char* name) {
    for (int i = 0; i < kNVariants; ++i)
        if (strcmp(name, kVariants[i].name) == 0)
            return i;
    return -1;
}

int tfc4_gemm(int variant, const void* A, const void* SA,
              const void* B, const void* SB, void* D,
              int64_t M, int64_t N, int64_t K, float alpha,
              cudaStream_t stream) {
    if (variant < 0 || variant >= kNVariants) {
        fprintf(stderr, "thor_fp4 cutlass: bad variant %d (0..%d)\n", variant, kNVariants - 1);
        return -1;
    }
    if (K % 64 != 0) {
        fprintf(stderr, "thor_fp4 cutlass: K must be a multiple of 64 (got %ld)\n", (long)K);
        return -3;
    }
    return kVariants[variant].fn(M, N, K, A, SA, B, SB, D, alpha, stream);
}

int tfc4_cutlass_version() {
    return CUTLASS_MAJOR * 10000 + CUTLASS_MINOR * 100 + CUTLASS_PATCH;
}

}  // extern "C"
