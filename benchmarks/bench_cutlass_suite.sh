#!/bin/bash
# bench_cutlass_suite.sh — CUTLASS variant measurements for BENCHMARKS.md:
#   1. big-GEMM tuning table at 8192^3 (all variants)
#   2. the small-K landscape: can our own kernels close the small-K gap
#      where cuBLASLt's FP4 kernels lose to BF16? (M=4096, the shapes
#      bench_gemm's section B measured for the cuBLASLt path)
#   3. direct cuBLASLt-vs-CUTLASS cross-check at one big shape through the
#      library (both paths already validate vs independent references; this
#      reports their mutual agreement)
set -e
cd "$(dirname "$0")/.."
mkdir -p benchmarks/results
OUT=benchmarks/results/cutlass.txt
{
  echo "# CUTLASS suite  date=$(date '+%F %T')  load=$(cut -d' ' -f1 /proc/loadavg)  mem_avail=$(free -g | awk '/Mem:/{print $7}')GiB"
  echo
  echo "== all variants at 8192^3 (correctness vs FP64 host ref + timing) =="
  ./build/bench_cutlass 8192 8192 8192 10
  echo
  echo "== small-K landscape M=4096 (does a tuned own kernel beat cuBLASLt's small-K?) =="
  for NK in "960 320" "2560 320" "1920 640" "320 2560" "5120 1280" "1280 5120"; do
    set -- $NK
    echo "-- N=$1 K=$2"
    ./build/bench_cutlass nvfp4_128x128x256_c1  4096 "$1" "$2" 20
    ./build/bench_cutlass nvfp4_128x128x64_c1   4096 "$1" "$2" 20
    ./build/bench_cutlass mxfp8xfp4_128x128x256_c1 4096 "$1" "$2" 20
  done
  echo
  echo "== mixed MXFP8xMXFP4 across shapes (the path cuBLASLt refuses) =="
  for MNK in "8192 8192 8192" "4096 5120 5120" "256 8192 8192" "4096 5120 1280"; do
    set -- $MNK
    ./build/bench_cutlass mxfp8xfp4_128x128x256_c1 "$1" "$2" "$3" 10
  done
} 2>&1 | tee "$OUT"

# cross-check: cuBLASLt vs CUTLASS on identical packed inputs, via the library
.venv/bin/python - <<'EOF' 2>&1 | tee -a benchmarks/results/cutlass.txt
import torch, thor_fp4
from thor_fp4 import ops
torch.manual_seed(0)
M, N, K = 2048, 2048, 2048
x = torch.randn(M, K, device="cuda") * 0.5
w = torch.randn(N, K, device="cuda") * 0.3
a = thor_fp4.NVFP4Tensor.from_float(x, 1.0)
b = thor_fp4.NVFP4Tensor.from_float(w, 1.0)
out_lt = torch.empty(M, N, dtype=torch.float16, device="cuda")
ops.gemm(a.packed, a.sblk, 1.0, b.packed, b.sblk, 1.0, out_lt, M, N, K, "f16")
L = thor_fp4._ext.cutlass()
out_ct = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
rc = L.tfc4_gemm(L.tfc4_variant_index(b"nvfp4_128x128x256_c1"),
                 a.packed.data_ptr(), a.sblk.data_ptr(),
                 b.packed.data_ptr(), b.sblk.data_ptr(),
                 out_ct.data_ptr(), M, N, K, 1.0, thor_fp4._ext.stream())
assert rc == 0
d = (out_lt.float() - out_ct.float())
rel = (d.pow(2).mean().sqrt() / out_lt.float().pow(2).mean().sqrt()).item()
print(f"\n# cuBLASLt vs CUTLASS nvfp4 on identical packed inputs ({M}x{N}x{K}): relRMS={rel:.3e}  load", open('/proc/loadavg').read().split()[0])
EOF
