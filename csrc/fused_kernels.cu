// fused_kernels.cu — the fused-epilogue chain kernels (ROUND 2, DIRECTIONS §4).
//
// What these are: GEMMs whose epilogue does the whole inter-layer glue of a
// small decision MLP in one pass —
//
//     D_codes, D_sf = quantize_mx8( gelu( alpha * (A @ B^T) + bias ) )
//
// i.e. the bias add, the GELU, and the MXFP8 quantization of the activations
// (e4m3 codes + one UE8M0 scale per 32 outputs) happen inside the GEMM's
// epilogue, while the accumulators are still in registers. The written
// (codes, scales) pair is exactly the operand format the next GEMM's A side
// consumes, so an MLP runs as one kernel per layer with no separate quantize
// pass at all. Why that matters (measured, BENCHMARKS.md §2): at small K the
// two-pass chain pays a full read+write over the activations to quantize
// them; that pass, not the GEMM, dominates small-K layers.
//
// Mainloop: MXFP8 activations x MXFP4 weights (kind::mxf8f6f4) — the mix
// cuBLASLt refuses on sm_110; same operand encoding as cutlass_kernels.cu.
//
// Numerics of the generated scales: CUTLASS's SfDGen recipe stores
// UE8M0(ceil)(amax * norm / 448) (cvt.rp — round toward +infinity), while our
// standalone quantize_mx kernel rounds to the NEAREST power of two (ties up).
// Ceil never saturates the e4m3 codes (ours can clip the block max by up to
// ~34% of 448 when the pow2 tie rounds down); the two recipes therefore
// deliver slightly different operands on some blocks. Both are measured in
// BENCHMARKS.md §5; the fused path is the one that runs here.
//
// Constraints (can_implement + measured): K % 128 == 0, N % 32 == 0 (the SF
// store tiles), N % 16 == 0 (e4m3 D alignment), bias is required (the EVT
// reads the bias pointer unconditionally — pass zeros for bias-free layers).
//
// C ABI (prefix tfc4f_):
//   tfc4f_n_variants() / tfc4f_variant_name(i) / tfc4f_variant_index(name)
//   tfc4f_gemm(variant, A, SA, B, SB, bias, D, SFD, M, N, K, alpha, stream)
//       A (M,K) mxfp8 codes + SA scales, B (N,K) mxfp4 codes + SB scales;
//       D (M,N) e4m3 codes row-major; SFD = tfc4f_sfd_bytes(M,N) bytes in the
//       tcgen05 128x4 scale-tile layout (same atom as the A-side scales, so
//       (D, SFD) feeds the next GEMM directly).
//   tfc4f_sfd_bytes(M, N)   exact cosize of the SFD layout
//
// Build: nvcc -arch=sm_110a -O2 -std=c++17 --expt-relaxed-constexpr
//        -Xcompiler -fPIC -shared -I$CUTLASS/include -I$CUTLASS/tools/util/include
//        -o build/libthorfp4_fused.so csrc/fused_kernels.cu -lcublasLt

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

// one fused variant: Tile x Cluster; A = MXFP8, B = MXFP4, D = MXFP8 codes
// + UE8M0 scales, epilogue = bias + GELU + scale-factor generation.
template <class Tile, class Cluster>
struct VTF {
    using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using ElementD = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementC = cutlass::bfloat16_t;
    using ElementAcc = float;
    using ArchTag = cutlass::arch::Sm100;   // compiled for sm_110a; see patches/
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
        cutlass::epilogue::thread::GELU, 32,
        ElementD, float, cutlass::float_ue8m0_t,
        cutlass::layout::RowMajor,   // scalefactor layout tag
        float,                       // ElementBias (fp32, like thor_fp4.Linear)
        ElementC, float, 4>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OpClass, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        ElementC, cutlass::layout::RowMajor, 8,
        ElementD, cutlass::layout::RowMajor, 16,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        ElementA, cutlass::layout::RowMajor, 32,
        ElementB, cutlass::layout::ColumnMajor, 128,
        ElementAcc, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// the epilogue reads norm_constant (== 1.0: the plain MX recipe) from device
// memory at kernel setup; allocate the constant once
const float* norm_one() {
    static float* p = nullptr;
    if (!p) {
        if (cudaMalloc(&p, 4) != cudaSuccess) return nullptr;
        float one = 1.0f;
        if (cudaMemcpy(p, &one, 4, cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    }
    return p;
}

// resolve+cache one (variant, M, N, K); per-call only pointers are updated
template <class V>
int ensure(int64_t M, int64_t N, int64_t K,
           const void* A, const void* SA, const void* B, const void* SB,
           const float* bias, void* D, void* SFD, float alpha,
           cudaStream_t stream) {
    using Gemm = typename V::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using LayoutSFD = cutlass::detail::Sm1xxBlockScaledOutputConfig<32>::LayoutSF;

    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    static std::map<std::tuple<int64_t, int64_t, int64_t>,
                    std::pair<std::shared_ptr<Gemm>, std::shared_ptr<void>>> cache;
    auto key = std::make_tuple(M, N, K);
    auto mkargs = [&](const void* a, const void* sa, const void* b, const void* sb,
                      const float* bi, void* d, void* sfd, float alpha) {
        typename Gemm::GemmKernel::Arguments args{};
        args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        args.problem_shape = {int(M), int(N), int(K), 1};
        StrideA stA = cutlass::make_cute_packed_stride(StrideA{}, {int(M), int(K), 1});
        StrideB stB = cutlass::make_cute_packed_stride(StrideB{}, {int(N), int(K), 1});
        auto lSFA = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFA(cute::make_shape(int(M), int(N), int(K), 1));
        auto lSFB = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
            tile_atom_to_shape_SFB(cute::make_shape(int(M), int(N), int(K), 1));
        args.mainloop = {
            reinterpret_cast<const typename V::ElementA::DataType*>(a), stA,
            reinterpret_cast<const typename V::ElementB::DataType*>(b), stB,
            reinterpret_cast<const typename V::ElementA::ScaleFactorType*>(sa), lSFA,
            reinterpret_cast<const typename V::ElementB::ScaleFactorType*>(sb), lSFB};
        StrideC stC = cutlass::make_cute_packed_stride(StrideC{}, {int(M), int(N), 1});
        StrideD stD = cutlass::make_cute_packed_stride(StrideD{}, {int(M), int(N), 1});
        auto& e = args.epilogue;
        e = {{}, nullptr, stC,
             reinterpret_cast<typename V::ElementD::DataType*>(d), stD};
        e.thread.alpha = alpha;
        e.thread.bias_ptr = bi;
        e.thread.block_scale_factor_ptr =
            reinterpret_cast<cutlass::float_ue8m0_t*>(sfd);
        e.thread.norm_constant_ptr = norm_one();
        return args;
    };
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto gemm = std::make_shared<Gemm>();
        const void* dmy = (const void*)0x10000ull;
        typename Gemm::GemmKernel::Arguments probe = mkargs(dmy, dmy, dmy, dmy,
                                                            (const float*)dmy,
                                                            (void*)dmy, (void*)dmy, alpha);
        if (gemm->can_implement(probe) != cutlass::Status::kSuccess)
            return -20;
        size_t ws_bytes = Gemm::get_workspace_size(probe);
        void* ws = nullptr;
        if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess)
            return -21;
        auto wsg = std::shared_ptr<void>(ws, [](void* p) { if (p) cudaFree(p); });
        if (gemm->initialize(probe, ws) != cutlass::Status::kSuccess)
            return -22;
        it = cache.emplace(key, std::make_pair(gemm, wsg)).first;
    }
    typename Gemm::GemmKernel::Arguments args = mkargs(A, SA, B, SB, bias, D, SFD, alpha);
    if (it->second.first->update(args) != cutlass::Status::kSuccess)
        return -23;
    if (it->second.first->run(stream) != cutlass::Status::kSuccess)
        return -24;
    return 0;
}

struct FVariant {
    const char* name;
    int (*fn)(int64_t, int64_t, int64_t, const void*, const void*,
              const void*, const void*, const float*, void*, void*, float,
              cudaStream_t);
};

const FVariant kFVariants[] = {
    {"mx8out_gelu_128x128x256_c1",
     &ensure<VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>>>},
    {"mx8out_gelu_128x128x256_c2x1",
     &ensure<VTF<Shape<_128, _128, _256>, Shape<_2, _1, _1>>>},
    {"mx8out_gelu_128x256x256_c2x1",
     &ensure<VTF<Shape<_128, _256, _256>, Shape<_2, _1, _1>>>},
    {"mx8out_gelu_128x128x128_c2x1",
     &ensure<VTF<Shape<_128, _128, _128>, Shape<_2, _1, _1>>>},
};
constexpr int kNFVariants = int(sizeof(kFVariants) / sizeof(kFVariants[0]));

}  // namespace

extern "C" {

int tfc4f_n_variants() { return kNFVariants; }

const char* tfc4f_variant_name(int i) {
    return (i >= 0 && i < kNFVariants) ? kFVariants[i].name : nullptr;
}

int tfc4f_variant_index(const char* name) {
    for (int i = 0; i < kNFVariants; ++i)
        if (strcmp(name, kFVariants[i].name) == 0)
            return i;
    return -1;
}

int64_t tfc4f_sfd_bytes(int64_t M, int64_t N) {
    auto layout = cutlass::detail::Sm1xxBlockScaledOutputConfig<32>::
        tile_atom_to_shape_SFD(cute::make_shape(int(M), int(N), 1, 1));
    return int64_t(cute::cosize(layout));
}

int tfc4f_gemm(int variant, const void* A, const void* SA,
               const void* B, const void* SB, const float* bias,
               void* D, void* SFD,
               int64_t M, int64_t N, int64_t K, float alpha,
               cudaStream_t stream) {
    if (variant < 0 || variant >= kNFVariants) {
        fprintf(stderr, "thor_fp4 fused: bad variant %d (0..%d)\n", variant, kNFVariants - 1);
        return -1;
    }
    if (K % 128 != 0) {
        fprintf(stderr, "thor_fp4 fused: K must be a multiple of 128 (got %ld)\n", (long)K);
        return -3;
    }
    if (N % 32 != 0) {
        fprintf(stderr, "thor_fp4 fused: N must be a multiple of 32 (got %ld)\n", (long)N);
        return -4;
    }
    if (!bias) {
        fprintf(stderr, "thor_fp4 fused: bias pointer is required (pass zeros)\n");
        return -5;
    }
    return kFVariants[variant].fn(M, N, K, A, SA, B, SB, bias, D, SFD, alpha, stream);
}

}  // extern "C"
