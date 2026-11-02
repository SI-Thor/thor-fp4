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
// reads the bias pointer unconditionally — pass zeros for bias-free layers)
// and is an N-vector indexed by output column n: D[m,n] += bias[n].
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

// one fused variant: Tile x Cluster x Activation; A = MXFP8, B = MXFP4,
// D = MXFP8 codes + UE8M0 scales, epilogue = bias + activation + scale-factor
// generation. GELU serves plain MLPs; SiLU the gated (SwiGLU-class) ones;
// Identity the un-activated halves of a gate.
template <class Tile, class Cluster,
          template <class> class Activation = cutlass::epilogue::thread::GELU>
struct VTF {
    using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    // D is the RAW e4m3 (example 92's pattern: the scaled-type wrappers are
    // for mainloop operands; the epilogue's TMA store takes the raw type)
    using ElementD = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementAcc = float;
    using ArchTag = cutlass::arch::Sm100;   // compiled for sm_110a; see patches/
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

    // PerColBias = a bias VECTOR INDEXED BY OUTPUT COLUMN n (D[m,n] gets
    // bias[n]) — nn.Linear semantics. The PerRowBias op we used through
    // ROUND 2 reads bias[m]: semantically wrong for nn.Linear (masked in
    // tests by zero-initialized biases) and, for M > N, an out-of-bounds
    // read of the N-float bias buffer — the source of the ue8m0-NaN rows
    // (BENCHMARKS sec.10 caveat; root-caused 2026-11-01, see NOTES.md).
    using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
        Activation, 32,
        ElementD, float, cutlass::float_ue8m0_t,
        cutlass::layout::RowMajor,   // scalefactor layout tag (the row SF store)
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
             reinterpret_cast<cutlass::float_e4m3_t*>(d), stD};
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
        // initialize (which also sets up any scheduler workspace) with the
        // REAL arguments — a probe-initialized workspace leaves stale state
        // that corrupted one row on the first run (measured: one row of a
        // 2048x1024x512 call decoded as ue8m0 NaN scales, deterministic)
        typename Gemm::GemmKernel::Arguments real = mkargs(A, SA, B, SB, bias, D, SFD, alpha);
        if (gemm->initialize(real, ws) != cutlass::Status::kSuccess)
            return -22;
        it = cache.emplace(key, std::make_pair(gemm, wsg)).first;
        if (it->second.first->run(stream) != cutlass::Status::kSuccess)
            return -24;
        return 0;
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
    // R3-x: SwiGLU-class activations — SiLU for the gated half, Identity
    // for the un-gated half of a gate, same PerColBias/SFD epilogue.
    {"mx8out_silu_128x128x256_c1",
     &ensure<VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>,
                 cutlass::epilogue::thread::SiLu>>},
    {"mx8out_silu_128x256x256_c2x1",
     &ensure<VTF<Shape<_128, _256, _256>, Shape<_2, _1, _1>,
                 cutlass::epilogue::thread::SiLu>>},
    {"mx8out_ident_128x128x256_c1",
     &ensure<VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>,
                 cutlass::epilogue::thread::Identity>>},
    {"mx8out_ident_128x256x256_c2x1",
     &ensure<VTF<Shape<_128, _256, _256>, Shape<_2, _1, _1>,
                 cutlass::epilogue::thread::Identity>>},
};
constexpr int kNFVariants = int(sizeof(kFVariants) / sizeof(kFVariants[0]));

// R3-0 bisection helpers: run one GEMM with per-call lifecycle variants so the
// cached ensure() path can be compared against a fresh object / re-initialized
// workspace / re-initialized workspace with a fresh allocation.
enum RunMode { kCached = 0, kReinitWs = 1, kFreshObject = 2, kFreshWs = 3 };

template <class V>
int run_mode(int mode, int64_t M, int64_t N, int64_t K,
             const void* A, const void* SA, const void* B, const void* SB,
             const float* bias, void* D, void* SFD, float alpha,
             cudaStream_t stream) {
    using Gemm = typename V::Gemm;
    if (mode == kCached)
        return ensure<V>(M, N, K, A, SA, B, SB, bias, D, SFD, alpha, stream);
    // build arguments the same way ensure() does
    typename Gemm::GemmKernel::Arguments args{};
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = {int(M), int(N), int(K), 1};
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stA = cutlass::make_cute_packed_stride(StrideA{}, {int(M), int(K), 1});
    StrideB stB = cutlass::make_cute_packed_stride(StrideB{}, {int(N), int(K), 1});
    auto lSFA = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
        tile_atom_to_shape_SFA(cute::make_shape(int(M), int(N), int(K), 1));
    auto lSFB = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig::
        tile_atom_to_shape_SFB(cute::make_shape(int(M), int(N), int(K), 1));
    args.mainloop = {
        reinterpret_cast<const typename V::ElementA::DataType*>(A), stA,
        reinterpret_cast<const typename V::ElementB::DataType*>(B), stB,
        reinterpret_cast<const typename V::ElementA::ScaleFactorType*>(SA), lSFA,
        reinterpret_cast<const typename V::ElementB::ScaleFactorType*>(SB), lSFB};
    StrideC stC = cutlass::make_cute_packed_stride(StrideC{}, {int(M), int(N), 1});
    StrideD stD = cutlass::make_cute_packed_stride(StrideD{}, {int(M), int(N), 1});
    args.epilogue = {{}, nullptr, stC,
                     reinterpret_cast<cutlass::float_e4m3_t*>(D), stD};
    args.epilogue.thread.alpha = alpha;
    args.epilogue.thread.bias_ptr = bias;
    args.epilogue.thread.block_scale_factor_ptr =
        reinterpret_cast<cutlass::float_ue8m0_t*>(SFD);
    args.epilogue.thread.norm_constant_ptr = norm_one();

    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (mode == kReinitWs) {
        // cached object, but initialize() again before every run
        static std::map<std::tuple<int64_t, int64_t, int64_t>,
                        std::pair<std::shared_ptr<Gemm>, std::shared_ptr<void>>> cache;
        auto key = std::make_tuple(M, N, K);
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto gemm = std::make_shared<Gemm>();
            size_t ws_bytes = Gemm::get_workspace_size(args);
            void* ws = nullptr;
            if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess) return -31;
            auto wsg = std::shared_ptr<void>(ws, [](void* p) { if (p) cudaFree(p); });
            if (gemm->initialize(args, ws) != cutlass::Status::kSuccess) return -32;
            it = cache.emplace(key, std::make_pair(gemm, wsg)).first;
        } else {
            if (it->second.first->initialize(args, it->second.second.get())
                != cutlass::Status::kSuccess)
                return -33;
        }
        return it->second.first->run(stream) == cutlass::Status::kSuccess ? 0 : -34;
    }
    if (mode == kFreshWs) {
        // cached object, but a FRESH cudaMalloc'd workspace + initialize per run
        static std::map<std::tuple<int64_t, int64_t, int64_t>,
                        std::shared_ptr<Gemm>> cache;
        auto key = std::make_tuple(M, N, K);
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto gemm = std::make_shared<Gemm>();
            it = cache.emplace(key, gemm).first;
        }
        size_t ws_bytes = Gemm::get_workspace_size(args);
        void* ws = nullptr;
        if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess) return -35;
        auto wsg = std::shared_ptr<void>(ws, [](void* p) { if (p) cudaFree(p); });
        if (it->second->initialize(args, ws) != cutlass::Status::kSuccess) return -36;
        int rc = it->second->run(stream) == cutlass::Status::kSuccess ? 0 : -37;
        return rc;
    }
    // kFreshObject: construct, initialize, run, destroy — nothing persists
    {
        Gemm gemm;
        size_t ws_bytes = Gemm::get_workspace_size(args);
        void* ws = nullptr;
        if (ws_bytes && cudaMalloc(&ws, ws_bytes) != cudaSuccess) return -38;
        if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) { if (ws) cudaFree(ws); return -39; }
        int rc = gemm.run(stream) == cutlass::Status::kSuccess ? 0 : -40;
        if (ws) cudaFree(ws);
        return rc;
    }
}

struct RMVariant {
    const char* name;
    int (*fn)(int, int64_t, int64_t, int64_t, const void*, const void*,
              const void*, const void*, const float*, void*, void*, float,
              cudaStream_t);
};

const RMVariant kRMVariants[] = {
    {"mx8out_gelu_128x128x256_c1",
     &run_mode<VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>>>},
    {"mx8out_gelu_128x128x256_c2x1",
     &run_mode<VTF<Shape<_128, _128, _256>, Shape<_2, _1, _1>>>},
    {"mx8out_gelu_128x256x256_c2x1",
     &run_mode<VTF<Shape<_128, _256, _256>, Shape<_2, _1, _1>>>},
    {"mx8out_gelu_128x128x128_c2x1",
     &run_mode<VTF<Shape<_128, _128, _128>, Shape<_2, _1, _1>>>},
    {"mx8out_silu_128x128x256_c1",
     &run_mode<VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>,
                   cutlass::epilogue::thread::SiLu>>},
    {"mx8out_silu_128x256x256_c2x1",
     &run_mode<VTF<Shape<_128, _256, _256>, Shape<_2, _1, _1>,
                   cutlass::epilogue::thread::SiLu>>},
    {"mx8out_ident_128x128x256_c1",
     &run_mode<VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>,
                   cutlass::epilogue::thread::Identity>>},
    {"mx8out_ident_128x256x256_c2x1",
     &run_mode<VTF<Shape<_128, _256, _256>, Shape<_2, _1, _1>,
                   cutlass::epilogue::thread::Identity>>},
};
constexpr int kNRMVariants = int(sizeof(kRMVariants) / sizeof(kRMVariants[0]));

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

// R3-0 bisection: the same GEMM with per-call lifecycle variants.
//   mode 0 = cached ensure() (the library path)
//   mode 1 = cached object, initialize() re-run before every launch
//   mode 2 = fresh Gemm object + fresh workspace every call
//   mode 3 = cached object, fresh cudaMalloc'd workspace + initialize every call
int tfc4f_gemm_mode(int variant, int mode, const void* A, const void* SA,
                    const void* B, const void* SB, const float* bias,
                    void* D, void* SFD,
                    int64_t M, int64_t N, int64_t K, float alpha,
                    cudaStream_t stream) {
    if (variant < 0 || variant >= kNRMVariants) return -1;
    return kRMVariants[variant].fn(mode, M, N, K, A, SA, B, SB, bias, D, SFD,
                                   alpha, stream);
}

int64_t tfc4f_ws_bytes(int variant, int64_t M, int64_t N, int64_t K) {
    // workspace bytes the (cached) Gemm object uses for this shape
    using Gemm = typename VTF<Shape<_128, _128, _256>, Shape<_1, _1, _1>>::Gemm;
    typename Gemm::GemmKernel::Arguments args{};
    args.problem_shape = {int(M), int(N), int(K), 1};
    return int64_t(Gemm::get_workspace_size(args));
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
