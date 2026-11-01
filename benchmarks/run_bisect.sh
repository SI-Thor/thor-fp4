#!/bin/bash
# run_bisect.sh — R3-0 helper: converter probe + deep repro + ABI paths.
set -u
cd "$(dirname "$0")/.."
NVCC=/usr/local/cuda-13.2/bin/nvcc
CUT=../lab-fp4/vendor/cutlass
export PYTHONUNBUFFERED=1
echo "=== nvcc probe2 build ==="
$NVCC -arch=sm_110a -O2 -std=c++17 -I$CUT/include \
    -o build/ue8m0_probe2 benchmarks/ue8m0_probe2.cu 2>&1 | tail -8
if [ -x build/ue8m0_probe2 ]; then
  echo "=== ue8m0 probe2 (CUTLASS converters on edge inputs) ==="
  ./build/ue8m0_probe2
fi
echo "=== repro2 deep ==="
.venv/bin/python benchmarks/repro_nan2.py
echo "=== abi paths M=2048 ==="
.venv/bin/python benchmarks/repro_nan.py --paths abi-reuse abi-fresh abi-cuda --M 2048 --calls 3 --sentinel
