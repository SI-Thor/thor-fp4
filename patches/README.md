# patches/

`cutlass-4.4.1-sm110.patch` enables the Thor's GPU (sm_110 / GB10) in
CUTLASS 4.4.1. CUTLASS has no sm_110 builders, but `cute/arch/config.hpp`
already gates the tcgen05 block-scaled MMA kinds (kind::mxf4nvf4,
kind::mxf8f6f4) for `SM110A`, so the sm_100 block-scaled collective builders
compile and run for `-arch=sm_110a` once they accept an `arch::Sm110` tag.
The patch adds the tag and lets the sm_100 block-scaled builder take it.

Apply to a pristine CUTLASS 4.4.1 checkout:

    git clone --branch v4.4.1 --depth 1 https://github.com/NVIDIA/cutlass.git vendor/cutlass
    cd vendor/cutlass && git apply ../patches/cutlass-4.4.1-sm110.patch

or point `CUTLASS_ROOT` at any patched 4.4.1 tree. CUTLASS is BSD-3-clause
(vendor/cutlass/LICENSE.txt upstream); this repo does not vendor it.
