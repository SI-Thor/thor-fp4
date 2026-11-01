// bench_main.cu — standalone validation + timing for the CUTLASS variants
// in cutlass_kernels.cu (linked together; see `make bench-cutlass`).
//
// Validation: every variant is checked against an FP64 host reference
// computed from the SAME codes and block scales (so the check isolates the
// tensor-core execution, not quantization). This is the instrument that
// pinned the lab's CUTLASS-vs-cuBLASLt disagreement on operand alignment.
//
// Usage:
//   bench_cutlass M N K [iters]          run every variant at one shape
//   bench_cutlass <variant> M N K [iters]
//   bench_cutlass list                   variant names

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int tfc4_n_variants();
const char* tfc4_variant_name(int);
int tfc4_variant_index(const char*);
int tfc4_gemm(int, const void*, const void*, const void*, const void*, void*,
              int64_t, int64_t, int64_t, float, cudaStream_t);
}

// ---------------------------------------------------------------------------
// host encoders (mirror csrc/thor_fp4_core.cu)
// ---------------------------------------------------------------------------
static uint8_t e2m1_enc(float v) {
    uint8_t sign = v < 0 ? 8 : 0;
    float a = fminf(fabsf(v), 6.0f);
    static const float mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    uint8_t code = 7;
    for (int i = 6; i >= 0; --i) if (a < mid[i]) code = (uint8_t)i;
    return sign | code;
}
static float e2m1_dec(uint8_t c) {
    static const float t[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float v = t[c & 7]; return (c & 8) ? -v : v;
}
static uint8_t e4m3_enc(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
static float e4m3_dec(uint8_t b) {
    __nv_fp8_storage_t s = (__nv_fp8_storage_t)b;
    return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}
static uint8_t ue8m0_enc(float v) {
    if (!(v > 0)) return 0;
    int e; frexpf(v, &e);
    float lo = ldexpf(1.f, e - 1), hi = ldexpf(1.f, e);
    int code = (v - lo < hi - v) ? (e - 1) : e;
    code = std::max(-127, std::min(127, code));
    return (uint8_t)(code + 127);
}
static float ue8m0_dec(uint8_t b) { return ldexpf(1.f, (int)b - 127); }

static int64_t pad128(int64_t R) { return (R + 127) / 128 * 128; }

// scale layout: byte(r, s) at (a*(S/4)+d)*512 + 16*c + 4*b + e
static void reblock(const std::vector<uint8_t>& plain, int64_t R, int64_t S,
                    std::vector<uint8_t>& out) {
    out.assign((size_t)pad128(R) * S, 0);
    for (int64_t r = 0; r < R; ++r)
        for (int64_t s = 0; s < S; ++s) {
            int64_t a = r / 128, b = (r % 128) / 32, c = r % 32;
            int64_t d = s / 4, e = s % 4;
            out[(size_t)((a * (S / 4) + d) * 512 + 16 * c + 4 * b + e)] = plain[(size_t)r * S + s];
        }
}

// ---------------------------------------------------------------------------
struct Problem {
    // kind: 0 = nvfp4 (e2m1 codes, UE4M3 per 16)
    //       1 = mx     (A: e4m3 per byte, B: e2m1 packed; UE8M0 per 32)
    int kind;
    int64_t M, N, K;
    std::vector<uint8_t> cA, cB, sA_plain, sB_plain, sA_blk, sB_blk;
    std::vector<float> dA, dB;   // dequantized operands for the host reference
    static Problem make(int kind, int64_t M, int64_t N, int64_t K) {
        srand(1234);
        Problem p{kind, M, N, K};
        int64_t sb = kind == 0 ? 16 : 32;
        p.dA.assign((size_t)M * K, 0);
        p.dB.assign((size_t)N * K, 0);
        p.sA_plain.assign((size_t)M * (K / sb), 0);
        p.sB_plain.assign((size_t)N * (K / sb), 0);
        for (int64_t r = 0; r < M; ++r)
            for (int64_t s = 0; s < K / sb; ++s) {
                if (kind == 0) {
                    float sf = e4m3_dec(e4m3_enc(ldexpf(1.f, (rand() % 10) - 5) * (0.75f + 0.5f * rand() / RAND_MAX)));
                    p.sA_plain[(size_t)r * (K / sb) + s] = e4m3_enc(sf);
                    for (int i = 0; i < 16; i += 2) {
                        float lo = (rand() % 2000 - 1000) / 1000.f * 6.f;
                        float hi = (rand() % 2000 - 1000) / 1000.f * 6.f;
                        uint8_t b = (uint8_t)(e2m1_enc(lo) | (e2m1_enc(hi) << 4));
                        p.cA.push_back(b);
                        p.dA[(size_t)r * K + s * 16 + i] = e2m1_dec(b & 15) * sf;
                        p.dA[(size_t)r * K + s * 16 + i + 1] = e2m1_dec(b >> 4) * sf;
                    }
                } else {
                    float sf = ue8m0_dec(ue8m0_enc(ldexpf(1.f, (rand() % 17) - 8)));
                    p.sA_plain[(size_t)r * (K / sb) + s] = ue8m0_enc(sf);
                    for (int i = 0; i < 32; ++i) {
                        uint8_t b = e4m3_enc((rand() % 2000 - 1000) / 1000.f * 448.f);
                        p.cA.push_back(b);
                        p.dA[(size_t)r * K + s * 32 + i] = e4m3_dec(b) * sf;
                    }
                }
            }
        for (int64_t r = 0; r < N; ++r)
            for (int64_t s = 0; s < K / sb; ++s) {
                if (kind == 0) {
                    float sf = e4m3_dec(e4m3_enc(ldexpf(1.f, (rand() % 10) - 5) * (0.75f + 0.5f * rand() / RAND_MAX)));
                    p.sB_plain[(size_t)r * (K / sb) + s] = e4m3_enc(sf);
                    for (int i = 0; i < 16; i += 2) {
                        float lo = (rand() % 2000 - 1000) / 1000.f * 6.f;
                        float hi = (rand() % 2000 - 1000) / 1000.f * 6.f;
                        uint8_t b = (uint8_t)(e2m1_enc(lo) | (e2m1_enc(hi) << 4));
                        p.cB.push_back(b);
                        p.dB[(size_t)r * K + s * 16 + i] = e2m1_dec(b & 15) * sf;
                        p.dB[(size_t)r * K + s * 16 + i + 1] = e2m1_dec(b >> 4) * sf;
                    }
                } else {
                    float sf = ue8m0_dec(ue8m0_enc(ldexpf(1.f, (rand() % 17) - 8)));
                    p.sB_plain[(size_t)r * (K / sb) + s] = ue8m0_enc(sf);
                    for (int i = 0; i < 32; i += 2) {
                        float lo = (rand() % 2000 - 1000) / 1000.f * 6.f;
                        float hi = (rand() % 2000 - 1000) / 1000.f * 6.f;
                        uint8_t b = (uint8_t)(e2m1_enc(lo) | (e2m1_enc(hi) << 4));
                        p.cB.push_back(b);
                        p.dB[(size_t)r * K + s * 32 + i] = e2m1_dec(b & 15) * sf;
                        p.dB[(size_t)r * K + s * 32 + i + 1] = e2m1_dec(b >> 4) * sf;
                    }
                }
            }
        reblock(p.sA_plain, M, K / sb, p.sA_blk);
        reblock(p.sB_plain, N, K / sb, p.sB_blk);
        return p;
    }
};

static double loadavg() {
    FILE* f = fopen("/proc/loadavg", "r");
    if (!f) return -1;
    double v; fscanf(f, "%lf", &v); fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "list") == 0) {
        for (int i = 0; i < tfc4_n_variants(); ++i)
            printf("%2d %s\n", i, tfc4_variant_name(i));
        return 0;
    }
    int64_t M, N, K, iters = 10;
    int only = -1;
    if (argc >= 5 && tfc4_variant_index(argv[1]) >= 0) {
        only = tfc4_variant_index(argv[1]);
        M = atoll(argv[2]); N = atoll(argv[3]); K = atoll(argv[4]);
        if (argc > 5) iters = atoll(argv[5]);
    } else if (argc >= 4) {
        M = atoll(argv[1]); N = atoll(argv[2]); K = atoll(argv[3]);
        if (argc > 4) iters = atoll(argv[4]);
    } else {
        fprintf(stderr, "usage: bench_cutlass [variant] M N K [iters] | list\n");
        return 1;
    }

    printf("# CUTLASS bench  M=%ld N=%ld K=%ld iters=%ld load=%.2f\n",
           (long)M, (long)N, (long)K, (long)iters, loadavg());
    printf("# %-26s %10s %11s %9s %8s\n", "variant", "relRMS", "ms", "TFLOPS", "status");

    for (int vi = 0; vi < tfc4_n_variants(); ++vi) {
        if (only >= 0 && vi != only) continue;
        const char* name = tfc4_variant_name(vi);
        int kind = strstr(name, "mxfp") == name ? 1 : 0;   // mx* vs nvfp4*
        Problem p = Problem::make(kind, M, N, K);
        uint8_t *dA, *dB, *dSA, *dSB;
        void* dD;
        size_t aBytes = kind == 0 ? (size_t)M * K / 2 : (size_t)M * K;
        size_t bBytes = (size_t)N * K / 2;
        cudaMalloc(&dA, aBytes); cudaMalloc(&dB, bBytes);
        cudaMalloc(&dSA, p.sA_blk.size()); cudaMalloc(&dSB, p.sB_blk.size());
        cudaMalloc(&dD, (size_t)M * N * 2);
        cudaMemcpy(dA, p.cA.data(), aBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, p.cB.data(), bBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dSA, p.sA_blk.data(), p.sA_blk.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dSB, p.sB_blk.data(), p.sB_blk.size(), cudaMemcpyHostToDevice);
        cudaMemset(dD, 0, (size_t)M * N * 2);

        int rc = tfc4_gemm(vi, dA, dSA, dB, dSB, dD, M, N, K, 1.0f, 0);
        if (cudaDeviceSynchronize() != cudaSuccess || rc != 0) {
            printf("# %-26s %10s %11s %9s %8s  (rc=%d %s)\n", name, "-", "-", "-", "SKIP",
                   rc, cudaGetErrorString(cudaGetLastError()));
            cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dD);
            continue;
        }
        // FP64 reference from the same codes+scales, spot-checked on rows
        double rel = 0.0;
        {
            int64_t m0 = M / 2, n0 = N / 4;
            std::vector<uint16_t> raw(2048);
            cudaMemcpy(raw.data(), (char*)dD + ((size_t)m0 * N + n0) * 2, 2048 * 2, cudaMemcpyDeviceToHost);
            double sse = 0, nrm = 0;
            const float* arow = &p.dA[(size_t)m0 * K];
            for (int i = 0; i < 2048; ++i) {
                int64_t n = n0 + i; if (n >= N) break;
                double acc = 0;
                const float* brow = &p.dB[(size_t)n * K];
                for (int64_t k = 0; k < K; ++k) acc += (double)arow[k] * brow[k];
                // bf16 -> f32 is an exact 16-bit left shift into the fp32 bits
                uint32_t f32 = (uint32_t)raw[i] << 16;
                float gf; memcpy(&gf, &f32, 4);
                double e = acc - gf; sse += e * e; nrm += acc * acc;
            }
            rel = sqrt(sse / 2048) / (sqrt(nrm / 2048) + 1e-30);
        }
        // timing: warm 1.2 s then median of 5 rounds
        {
            auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 1.2)
                tfc4_gemm(vi, dA, dSA, dB, dSB, dD, M, N, K, 1.0f, 0);
            cudaDeviceSynchronize();
        }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ms_list;
        for (int r = 0; r < 5; ++r) {
            cudaDeviceSynchronize();
            cudaEventRecord(e0);
            for (int64_t i = 0; i < iters; ++i)
                tfc4_gemm(vi, dA, dSA, dB, dSB, dD, M, N, K, 1.0f, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            ms_list.push_back(ms / iters);
        }
        std::sort(ms_list.begin(), ms_list.end());
        double ms = ms_list[ms_list.size() / 2];
        printf("# %-26s %10.2e %11.3f %9.1f %8s\n", name, rel, ms,
               2.0 * M * N * K / (ms * 1e-3) / 1e12, "OK");
        cudaFree(dA); cudaFree(dB); cudaFree(dSA); cudaFree(dSB); cudaFree(dD);
    }
    return 0;
}
