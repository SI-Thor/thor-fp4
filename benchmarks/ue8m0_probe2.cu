// ue8m0_probe2.cu — R3-0: the exact converters the SfDGen epilogue uses,
// called on edge inputs. Uses CUTLASS's own NumericConverter / NumericArray-
// Converter (PTX cvt.rp.satfinite.ue8m0x2.f32 on sm_110a) so the probe tests
// the production code path, not a hand-copied instruction.
//   nvcc -arch=sm_110a -O2 -std=c++17 -I<CUTLASS>/include \
//        -o build/ue8m0_probe2 benchmarks/ue8m0_probe2.cu
#include <cstdio>

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

using namespace cutlass;

__global__ void probe_scalar(const float* in, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    NumericConverter<float_ue8m0_t, float, FloatRoundStyle::round_toward_infinity> cvt;
    out[i] = cvt(in[i]).storage;
}

__global__ void probe_arr4(const float* in, unsigned char* out, int n4) {
    // the packed-4 elementwise converter SfDGen uses for the scale store
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    Array<float, 4> src;
    for (int j = 0; j < 4; j++) src[j] = in[i * 4 + j];
    NumericArrayConverter<float_ue8m0_t, float, 4,
                          FloatRoundStyle::round_toward_infinity> cvt;
    Array<float_ue8m0_t, 4> r = cvt(src);
    for (int j = 0; j < 4; j++) out[i * 4 + j] = r[j].storage;
}

__global__ void probe_back(const unsigned char* in, float* out, int n) {
    // ue8m0 byte -> float (the rcp/decode side the next GEMM sees)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float_ue8m0_t v;
    v.storage = in[i];
    NumericConverter<float, float_ue8m0_t> cvt;
    out[i] = cvt(v);
}

int main() {
    const float in[] = {
        0.0f, -0.0f, 1.4e-45f, 1e-40f, 7.5e-39f, 2.9387359e-39f, 2.9387359e-39f * 0.5f,
        2.0f / 448.0f, 1.0f / 448.0f, 1e-30f, 2e-23f, 2.0f, -2.0f, -1e-30f, -1e-40f,
        448.0f, 3.4e38f, 1.7e38f, 5.9e46f, 1.19e77f, 2e127f,
        0.0f / 0.0f, -(0.0f / 0.0f), 1.0f / 0.0f, -(1.0f / 0.0f)
    };
    const int n = sizeof(in) / sizeof(in[0]);
    float* d_in; unsigned char* d_out; float* d_back;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, 4 * n);
    cudaMalloc(&d_back, n * sizeof(float));
    cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice);
    probe_scalar<<<(n + 31) / 32, 32>>>(d_in, d_out, n);
    probe_arr4<<<(n / 4 + 31) / 32, 32>>>(d_in, d_out + n, n / 4);
    unsigned char host[4 * n];
    cudaMemcpy(host, d_out, 4 * n, cudaMemcpyDeviceToHost);
    probe_back<<<(n + 31) / 32, 32>>>(d_out, d_back, n);
    float back[n];
    cudaMemcpy(back, d_back, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%-26s %-12s %-12s %s\n", "input", "scalar cvt", "packed4 cvt", "byte->float");
    for (int i = 0; i < n; i++) {
        float v = in[i];
        const char* vs = (v != v) ? "NaN" : nullptr;
        char buf[32];
        if (vs) snprintf(buf, sizeof buf, "%s", vs); else snprintf(buf, sizeof buf, "%g", v);
        printf("%-26s 0x%02x        0x%02x        %g%s\n", buf,
               host[i], host[n + i], back[i],
               host[i] == 0xFF ? "   <-- NaN encoding" : "");
    }
    return 0;
}
