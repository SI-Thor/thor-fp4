// thor_fp4_core.cu — the core NVFP4 ops of thor-fp4.
//
// What is in here (all device-side, all CUDA-graph-capture-safe after the
// first call of a given shape):
//   * NVFP4 quantization of f32/f16/bf16 row-major tensors into the buffers
//     the sm_110 tensor cores read: e2m1 codes packed 2/byte plus one UE4M3
//     scale per 16 elements in the tcgen05 128x4-tile layout, with an
//     optional per-tensor scale folded into the GEMM's alpha (the two-level
//     NVFP4 recipe).
//   * The cuBLASLt block-scaled GEMM (CUDA_R_4F_E2M1, VEC16_UE4M3).
//   * Dequantization and scale un-scattering, used by thor_fp4.verify()'s
//     independent reference path.
//   * MX quantization (e2m1 or e4m3 codes, UE8M0 scale per 32) feeding the
//     CUTLASS kernels in cutlass_kernels.cu.
//
// Layout notes (all measured on the Thor, sm_110, cuBLASLt 13.4.1 — see
// README.md "How the hardware wants its operands"):
//   * Only the TN operand order returns algorithms: A (K,rows) col-major ==
//     row-major (rows,K) buffers for both operands.
//   * The scale-block layout for logical scale (row r, sf column s), with
//     a=r/128, b=(r%128)/32, c=r%32, d=s/4, e=s%4 is
//         byte index = (a*(S/4) + d)*512 + 16*c + 4*b + e
//     (128-row x 4-sf tiles; matches CUTLASS SfKMajorAtom).
//   * K must be a multiple of 64; the GEMM's M (here: the weight row count)
//     must be >= 8 and a multiple of 8; batch rows are padded to >= 8 by the
//     caller; packed and scale buffers are padded to 128-row tiles and must
//     be zero-initialized (rows >= R are never written by the kernels).
//   * cuBLASLt returns a row-major-D algorithm that computes WRONG values
//     (CUBLASLT_ORDER_ROW, measured); we never use it. Instead of the
//     column-major D + transpose the lab used, we swap the operands:
//     out(M,N) = act @ W^T is computed as (W @ act^T)^T, i.e. A=W (K,N),
//     B=act (K,M), D col-major (N,M) — which is exactly out row-major (M,N).
//     Same TN layout the tensor cores want, zero transposes.
//
// C ABI (returns 0 on success; negative on error, see stderr):
//   tfp4_init(ws_bytes)                    cublasLt handle + workspace
//   tfp4_cublaslt_version()
//   tfp4_quant(x, R, K, in_dtype, ts, packed, sblk, stream)
//   tfp4_quant_mx(x, R, K, in_dtype, kind, packed, sblk, stream)
//   tfp4_gemm(pA, sA, tsA, pB, sB, tsB, out, M, N, K, out_dtype, stream)
//   tfp4_dequant_f32(packed, sblk, ts, R, K, out, stream)
//   tfp4_scales_f32(sblk, R, K, out, stream)
//   tfp4_pad_rows(R)                       rows a buffer must be sized for
//   tfp4_packed_bytes(R, K), tfp4_scale_bytes(R, K), tfp4_mx_scale_bytes(R, K)
//
// Build: nvcc -arch=sm_110 -O2 -std=c++17 -Xcompiler -fPIC -shared \
//          -o build/libthorfp4_core.so csrc/thor_fp4_core.cu -lcublasLt

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"thor_fp4 CUDA err %s @%d\n",cudaGetErrorString(e_),__LINE__); return -1;} } while(0)
#define CKL(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"thor_fp4 cublasLt status %d @%d\n",(int)s_,__LINE__); return -2;} } while(0)

namespace {

// ---------------------------------------------------------------------------
// encoders
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t e2m1_enc(float v) {
    uint8_t sign = v < 0.f ? 8 : 0;
    float a = fminf(fabsf(v), 6.0f);
    // midpoints of {0,.5,1,1.5,2,3,4,6}; ties round up, matching the torch
    // fake quantizer (bucketize right=True)
    uint8_t code;
    if      (a < 0.25f) code = 0;
    else if (a < 0.75f) code = 1;
    else if (a < 1.25f) code = 2;
    else if (a < 1.75f) code = 3;
    else if (a < 2.50f) code = 4;
    else if (a < 3.50f) code = 5;
    else if (a < 5.00f) code = 6;
    else                code = 7;
    return sign | code;
}
__device__ __forceinline__ uint8_t e4m3_enc_pos(float v) {
    __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return (uint8_t)(((uint8_t)s) & 0x7f);
}
__device__ __forceinline__ float e4m3_dec(uint8_t b) {
    __nv_fp8_storage_t s = (__nv_fp8_storage_t)b;
    return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}
__device__ __forceinline__ uint8_t e4m3_enc(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
// UE8M0: value = 2^(code-127), round to nearest power of two
__device__ __forceinline__ uint8_t ue8m0_enc(float v) {
    if (!(v > 0.f)) return 0;
    int e; frexpf(v, &e);
    float lo = ldexpf(1.f, e - 1), hi = ldexpf(1.f, e);
    int code = (v - lo < hi - v) ? (e - 1) : e;
    if (code < -127) code = -127;
    if (code > 127) code = 127;
    return (uint8_t)(code + 127);
}

__device__ __forceinline__ float2 half2_to_float2(__half2 h) {
    return __half22float2(h);
}

// ---------------------------------------------------------------------------
// NVFP4 quantize: one block per (row, segment of scale groups). Reads 16
// elements per scale group vectorized (4x float4 for f32, 2x uint4 for the
// 16-bit dtypes), emits packed codes and a plain (R, K/16) f32 scale array.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void quant_nv_plain(const T* __restrict__ x, int64_t K,
                               uint8_t* __restrict__ packed,
                               float* __restrict__ sflat, float ts) {
    int64_t r = blockIdx.x;
    int64_t S = K / 16;
    int64_t seg = blockIdx.y, segs = gridDim.y;
    int64_t per = (S + segs - 1) / segs;
    int64_t s0 = seg * per, s1 = min(S, s0 + per);
    const T* row = x + r * K;
    uint8_t* prow = packed + r * (K / 2);
    float* srow = sflat + r * S;
    for (int64_t s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        float f[16];
        if constexpr (sizeof(T) == 4) {
            const float4* blk = (const float4*)(row + s * 16);
            float4 q0 = blk[0], q1 = blk[1], q2 = blk[2], q3 = blk[3];
            const float4 vv[4] = {q0, q1, q2, q3};
#pragma unroll
            for (int i = 0; i < 4; ++i) { f[4*i] = vv[i].x; f[4*i+1] = vv[i].y; f[4*i+2] = vv[i].z; f[4*i+3] = vv[i].w; }
        } else {
            // 16-bit dtypes (half / __nv_bfloat16): 2x uint4 = 16 elements
            const uint4* blk = (const uint4*)(row + s * 16);
            uint4 w0 = blk[0], w1 = blk[1];
            const uint4 ww[2] = {w0, w1};
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const __half2* hh = (const __half2*)&ww[i];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 ff = half2_to_float2(hh[j]);
                    f[8*i + 2*j] = ff.x; f[8*i + 2*j + 1] = ff.y;
                }
            }
        }
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i]));
        // two-level recipe: with a per-tensor scale ts, block scales
        // amax/(6*ts) sit near the top of the UE4M3 range for wide-range
        // tensors; ts == 1 is the block-only scheme
        uint8_t sc = e4m3_enc_pos(amax * (1.0f / 6.0f) / ts);
        float ds = e4m3_dec(sc) * ts;
        srow[s] = ds;
        uint8_t* dst = prow + s * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t lo = e2m1_enc(f[2 * i] / ds);
            uint8_t hi = e2m1_enc(f[2 * i + 1] / ds);
            dst[i] = (uint8_t)(lo | (hi << 4));
        }
    }
}

// reblock the plain scales into the tensor-core 128x4-tile layout (see the
// layout note at the top); grid-stride over 16-sf groups per row
__global__ void reblock_scales(const float* __restrict__ sflat, uint8_t* __restrict__ sblk,
                               int64_t R, int64_t S) {
    int64_t D = S / 4;
    int64_t gpr = (S + 15) / 16;
    int64_t ngroups = R * gpr;
    for (int64_t gi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; gi < ngroups;
         gi += (int64_t)gridDim.x * blockDim.x) {
        int64_t r = gi / gpr, sg = (gi % gpr) * 16;
        int64_t a = r / 128, b = (r % 128) / 32, c = r % 32;
        uint8_t* base = sblk + (a * D) * 512 + 16 * c + 4 * b;
        const float* src = sflat + r * S + sg;
        for (int d = 0; d < 4; ++d) {
            int64_t s0 = sg + d * 4;
            if (s0 >= S) break;
            uint8_t* dst = base + (s0 / 4) * 512;
            dst[0] = e4m3_enc_pos(src[d * 4 + 0]);
            if (s0 + 1 < S) dst[1] = e4m3_enc_pos(src[d * 4 + 1]);
            if (s0 + 2 < S) dst[2] = e4m3_enc_pos(src[d * 4 + 2]);
            if (s0 + 3 < S) dst[3] = e4m3_enc_pos(src[d * 4 + 3]);
        }
    }
}

// ---------------------------------------------------------------------------
// MX quantize (e2m1 or e4m3 codes, one UE8M0 scale per 32 elements) — feeds
// the CUTLASS kernels; same tile atom with S = K/32 scale columns.
// kind 0: e2m1 packed 2/byte (MXFP4);  kind 1: e4m3 1/byte (MXFP8).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void quant_mx(const T* __restrict__ x, int64_t K, int kind,
                         uint8_t* __restrict__ packed, float* __restrict__ sflat) {
    int64_t r = blockIdx.x;
    int64_t S = K / 32;
    int64_t seg = blockIdx.y, segs = gridDim.y;
    int64_t per = (S + segs - 1) / segs;
    int64_t s0 = seg * per, s1 = min(S, s0 + per);
    const T* row = x + r * K;
    float* srow = sflat + r * S;
    float maxval = kind == 0 ? 6.0f : 448.0f;
    for (int64_t s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        float f[32];
        if constexpr (sizeof(T) == 4) {
            const float4* blk = (const float4*)(row + s * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float4 v = blk[i];
                f[4*i] = v.x; f[4*i+1] = v.y; f[4*i+2] = v.z; f[4*i+3] = v.w;
            }
        } else {
            const uint4* blk = (const uint4*)(row + s * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint4 w = blk[i];
                const __half2* hh = (const __half2*)&w;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 ff = half2_to_float2(hh[j]);
                    f[8*i + 2*j] = ff.x; f[8*i + 2*j + 1] = ff.y;
                }
            }
        }
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
        uint8_t scb = ue8m0_enc(amax / maxval);
        float ds = ldexpf(1.f, (int)scb - 127);
        srow[s] = ds;
        if (kind == 0) {
            uint8_t* dst = packed + r * (K / 2) + s * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                uint8_t lo = e2m1_enc(f[2 * i] / ds);
                uint8_t hi = e2m1_enc(f[2 * i + 1] / ds);
                dst[i] = (uint8_t)(lo | (hi << 4));
            }
        } else {
            uint8_t* dst = packed + r * K + s * 32;
#pragma unroll
            for (int i = 0; i < 32; ++i) dst[i] = e4m3_enc(f[i] / ds);
        }
    }
}

__global__ void reblock_scales_ue8m0(const float* __restrict__ sflat, uint8_t* __restrict__ sblk,
                                     int64_t R, int64_t S) {
    int64_t D = S / 4;
    int64_t gpr = (S + 15) / 16;
    int64_t ngroups = R * gpr;
    for (int64_t gi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; gi < ngroups;
         gi += (int64_t)gridDim.x * blockDim.x) {
        int64_t r = gi / gpr, sg = (gi % gpr) * 16;
        int64_t a = r / 128, b = (r % 128) / 32, c = r % 32;
        uint8_t* base = sblk + (a * D) * 512 + 16 * c + 4 * b;
        const float* src = sflat + r * S + sg;
        for (int d = 0; d < 4; ++d) {
            int64_t s0 = sg + d * 4;
            if (s0 >= S) break;
            uint8_t* dst = base + (s0 / 4) * 512;
            dst[0] = ue8m0_enc(src[d * 4 + 0]);
            if (s0 + 1 < S) dst[1] = ue8m0_enc(src[d * 4 + 1]);
            if (s0 + 2 < S) dst[2] = ue8m0_enc(src[d * 4 + 2]);
            if (s0 + 3 < S) dst[3] = ue8m0_enc(src[d * 4 + 3]);
        }
    }
}

// ---------------------------------------------------------------------------
// dequant (reference path for verify): blocked scales -> plain, then codes.
// out is f32 row-major (R, K); value = e2m1(code) * sf * ts
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t sf_index(int64_t r, int64_t s, int64_t S) {
    int64_t a = r / 128, b = (r % 128) / 32, c = r % 32;
    int64_t d = s / 4, e = s % 4;
    return (a * (S / 4) + d) * 512 + 16 * c + 4 * b + e;
}

__global__ void dequant_f32(const uint8_t* __restrict__ packed, const uint8_t* __restrict__ sblk,
                            float ts, int64_t R, int64_t K, float* __restrict__ out) {
    int64_t S = K / 16;
    int64_t total = R * S;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        int64_t r = i / S, s = i % S;
        float ds = e4m3_dec(sblk[sf_index(r, s, S)]) * ts;
        const uint8_t* src = packed + r * (K / 2) + s * 8;
        float* dst = out + r * K + s * 16;
        const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint8_t b = src[j];
            dst[2 * j]     = t[b & 7] * ((b & 8) ? -1.f : 1.f) * ds;
            dst[2 * j + 1] = t[(b >> 4) & 7] * ((b & 128) ? -1.f : 1.f) * ds;
        }
    }
}

__global__ void scales_f32(const uint8_t* __restrict__ sblk, int64_t R, int64_t K,
                           float* __restrict__ out) {
    int64_t S = K / 16;
    int64_t total = R * S;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        int64_t r = i / S, s = i % S;
        out[i] = e4m3_dec(sblk[sf_index(r, s, S)]);
    }
}

// ---------------------------------------------------------------------------
// global state
// ---------------------------------------------------------------------------
cublasLtHandle_t g_lt = nullptr;
void* g_ws = nullptr;
size_t g_ws_bytes = 32ull << 20;
float* g_sflat = nullptr;      // quantize scratch (plain scales), grown on demand
size_t g_sflat_elems = 0;
std::mutex g_mu;

int ensure_sflat(size_t elems) {
    if (g_sflat_elems >= elems) return 0;
    if (g_sflat) cudaFree(g_sflat);
    CK(cudaMalloc(&g_sflat, elems * 4));
    g_sflat_elems = elems;
    return 0;
}

}  // namespace

extern "C" {

int tfp4_init(size_t ws_bytes) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_lt) return 0;
    if (ws_bytes) g_ws_bytes = ws_bytes;
    CKL(cublasLtCreate(&g_lt));
    CK(cudaMalloc(&g_ws, g_ws_bytes));
    return 0;
}

size_t tfp4_cublaslt_version() { return cublasLtGetVersion(); }

int64_t tfp4_pad_rows(int64_t R) { return (R + 127) / 128 * 128; }
int64_t tfp4_packed_bytes(int64_t R, int64_t K) {
    return ((R + 127) / 128 * 128) * K / 2;
}
int64_t tfp4_scale_bytes(int64_t R, int64_t K) {
    int64_t S = K / 16;
    return ((R + 127) / 128 * 128) * ((S + 3) / 4 * 4);
}
int64_t tfp4_mx_packed_bytes(int64_t R, int64_t K, int kind) {
    int64_t per = kind == 0 ? K / 2 : K;
    return ((R + 127) / 128 * 128) * per;
}
int64_t tfp4_mx_scale_bytes(int64_t R, int64_t K) {
    int64_t S = K / 32;
    return ((R + 127) / 128 * 128) * ((S + 3) / 4 * 4);
}

// quantize a row-major (R,K) tensor (in_dtype 0=f32 1=f16 2=bf16) to NVFP4.
// ts is the per-tensor scale (1.0 = block scales only). Buffers must be
// pre-zeroed and padded (tfp4_packed_bytes / tfp4_scale_bytes).
int tfp4_quant(const void* x, int64_t R, int64_t K, int in_dtype, float ts,
               uint8_t* packed, uint8_t* sblk, cudaStream_t stream) {
    if ((K % 64) != 0) { fprintf(stderr, "thor_fp4: K must be a multiple of 64 (got %ld)\n", (long)K); return -3; }
    std::lock_guard<std::mutex> lk(g_mu);
    int rc = ensure_sflat((size_t)R * (K / 16));
    if (rc) return rc;
    int64_t S = K / 16;
    int threads = (int)std::min<int64_t>(S < 512 ? S : 512, 512);
    // spread small row counts over more blocks so the quantize pass is not
    // launch-limited at small batch
    int segs = (int)std::min<int64_t>((S + threads - 1) / threads, std::max<int64_t>(1, 512 / (R + 1) + 1));
    dim3 grid((unsigned)R, (unsigned)segs);
    if (in_dtype == 0)
        quant_nv_plain<float><<<grid, threads, 0, stream>>>((const float*)x, K, packed, g_sflat, ts);
    else if (in_dtype == 1)
        quant_nv_plain<__half><<<grid, threads, 0, stream>>>((const __half*)x, K, packed, g_sflat, ts);
    else
        quant_nv_plain<__nv_bfloat16><<<grid, threads, 0, stream>>>((const __nv_bfloat16*)x, K, packed, g_sflat, ts);
    CK(cudaGetLastError());
    reblock_scales<<<(unsigned)((R * ((S + 15) / 16) + 255) / 256), 256, 0, stream>>>(
        g_sflat, sblk, R, S);
    CK(cudaGetLastError());
    return 0;
}

// MX quantize: kind 0 = MXFP4 (e2m1), kind 1 = MXFP8 e4m3. UE8M0 per-32 scales.
int tfp4_quant_mx(const void* x, int64_t R, int64_t K, int in_dtype, int kind,
                  uint8_t* packed, uint8_t* sblk, cudaStream_t stream) {
    if ((K % 64) != 0) { fprintf(stderr, "thor_fp4: K must be a multiple of 64 (got %ld)\n", (long)K); return -3; }
    std::lock_guard<std::mutex> lk(g_mu);
    int rc = ensure_sflat((size_t)R * (K / 32));
    if (rc) return rc;
    int64_t S = K / 32;
    int threads = (int)std::min<int64_t>(S < 512 ? S : 512, 512);
    int segs = (int)std::min<int64_t>((S + threads - 1) / threads, std::max<int64_t>(1, 512 / (R + 1) + 1));
    dim3 grid((unsigned)R, (unsigned)segs);
    if (in_dtype == 0)
        quant_mx<float><<<grid, threads, 0, stream>>>((const float*)x, K, kind, packed, g_sflat);
    else if (in_dtype == 1)
        quant_mx<__half><<<grid, threads, 0, stream>>>((const __half*)x, K, kind, packed, g_sflat);
    else
        quant_mx<__nv_bfloat16><<<grid, threads, 0, stream>>>((const __nv_bfloat16*)x, K, kind, packed, g_sflat);
    CK(cudaGetLastError());
    reblock_scales_ue8m0<<<(unsigned)((R * ((S + 15) / 16) + 255) / 256), 256, 0, stream>>>(
        g_sflat, sblk, R, S);
    CK(cudaGetLastError());
    return 0;
}

// ---------------------------------------------------------------------------
// the GEMM. out row-major (M,N) = A(M,K) @ B(N,K)^T, both operands NVFP4
// buffers. out_dtype 0=f16 1=bf16 2=f32. Implemented as D' = B @ A^T with
// cuBLASLt TN (A'=B: layout (K,N); B'=A: layout (K,M); D' col-major (N,M)),
// which the col-major D' buffer makes identical to out row-major — no
// transpose kernel (see the layout note at the top).
//
// alpha = tsA * tsB applies the two-level per-tensor scales.
//
// Everything shape-dependent is resolved once per (M,N,K,out_dtype) and
// reused, so repeat calls (including under CUDA graph capture) only enqueue
// the matmul.
// ---------------------------------------------------------------------------
struct Resolved {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
};
static std::map<std::tuple<int64_t, int64_t, int64_t, int>, Resolved> g_resolved;

int tfp4_gemm(const void* packedA, const void* sblkA, float tsA,
              const void* packedB, const void* sblkB, float tsB,
              void* out, int64_t M, int64_t N, int64_t K, int out_dtype,
              cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_lt) {
        CKL(cublasLtCreate(&g_lt));
        CK(cudaMalloc(&g_ws, g_ws_bytes));
    }
    cudaDataType_t odt = out_dtype == 0 ? CUDA_R_16F : (out_dtype == 1 ? CUDA_R_16BF : CUDA_R_32F);
    auto key = std::make_tuple(M, N, K, out_dtype);
    auto it = g_resolved.find(key);
    if (it == g_resolved.end()) {
        Resolved r;
        CKL(cublasLtMatmulDescCreate(&r.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t tT = CUBLAS_OP_T, tN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSA, &tT, sizeof(tT));
        cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof(tN));
        int32_t sm = (int32_t)CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
        cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
        // the heuristic returns nothing unless the scale pointers are non-null;
        // set dummies now, real pointers per call
        const void* dummy = (const void*)0x100000ull;
        cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy, sizeof(dummy));
        cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy, sizeof(dummy));
        // A' = the weight operand B (K,N) col-major == (N,K) row-major buffer
        CKL(cublasLtMatrixLayoutCreate(&r.la, CUDA_R_4F_E2M1, K, N, K));
        // B' = the activation operand A (K,M) col-major == (M,K) row-major buffer
        CKL(cublasLtMatrixLayoutCreate(&r.lb, CUDA_R_4F_E2M1, K, M, K));
        // D' col-major (N,M): buffer[n + m*N] == out row-major (M,N)[m*N + n]
        CKL(cublasLtMatrixLayoutCreate(&r.lc, odt, N, M, N));
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &g_ws_bytes, sizeof(g_ws_bytes));
        cublasLtMatmulHeuristicResult_t res[8];
        int found = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_lt, r.op, r.la, r.lb, r.lc, r.lc,
                                                           pref, 8, res, &found);
        cublasLtMatmulPreferenceDestroy(pref);
        if (st != CUBLAS_STATUS_SUCCESS || found == 0) {
            fprintf(stderr, "thor_fp4: no cublasLt algorithm for M=%ld N=%ld K=%ld out=%d\n",
                    (long)M, (long)N, (long)K, out_dtype);
            return -4;
        }
        r.algo = res[0].algo;
        it = g_resolved.emplace(key, r).first;
    }
    Resolved& r = it->second;
    const void* pa = sblkB;  // A' is the weight operand
    const void* pb = sblkA;  // B' is the activation operand
    cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa, sizeof(pa));
    cublasLtMatmulDescSetAttribute(r.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb, sizeof(pb));
    float alpha = tsA * tsB, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(g_lt, r.op, &alpha, packedB, r.la, packedA, r.lb,
                                       &beta, out, r.lc, out, r.lc, &r.algo,
                                       g_ws, g_ws_bytes, stream);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "thor_fp4 matmul status %d\n", (int)st); return -5; }
    return 0;
}

// dequant NVFP4 -> f32 row-major (R,K): the reference operand builder for
// thor_fp4.verify's independent path
int tfp4_dequant_f32(const uint8_t* packed, const uint8_t* sblk, float ts,
                     int64_t R, int64_t K, float* out, cudaStream_t stream) {
    int64_t S = K / 16;
    int64_t total = R * S;
    dequant_f32<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(packed, sblk, ts, R, K, out);
    CK(cudaGetLastError());
    return 0;
}

// blocked UE4M3 scales -> plain f32 (R, K/16)
int tfp4_scales_f32(const uint8_t* sblk, int64_t R, int64_t K, float* out,
                    cudaStream_t stream) {
    int64_t S = K / 16;
    int64_t total = R * S;
    scales_f32<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(sblk, R, K, out);
    CK(cudaGetLastError());
    return 0;
}

}  // extern "C"
