#!/bin/bash
# fetch-cutlass.sh — obtain a patched CUTLASS 4.4.1 for `make cutlass`.
# Puts it in vendor/cutlass (not committed; BSD-3 upstream). Offline
# alternative: point CUTLASS_ROOT at any CUTLASS 4.4.1 tree that already has
# patches/cutlass-4.4.1-sm110.patch applied.
set -e
cd "$(dirname "$0")/.."
if [ -d vendor/cutlass/include ]; then
    echo "vendor/cutlass already present"
    exit 0
fi
mkdir -p vendor
git clone --branch v4.4.1 --depth 1 https://github.com/NVIDIA/cutlass.git vendor/cutlass
cd vendor/cutlass
git apply ../../patches/cutlass-4.4.1-sm110.patch
echo "CUTLASS 4.4.1 + sm_110 patch ready in vendor/cutlass"
