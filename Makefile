# thor-fp4 build. Everything is local: a .venv in this directory, a CUDA
# shared library in build/, no system installs.
#
#   make venv          create .venv and install torch/numpy/pytest (wheel cache)
#   make core          build build/libthorfp4_core.so   (cuBLASLt path; ~40 s)
#   make cutlass       build build/libthorfp4_cutlass.so (own sm_110a kernels; ~5-15 min,
#                      4-8 GB RAM -> run under `flock ~/lm-wt/heavy.lock` on a shared box)
#   make bench-cutlass standalone CUTLASS benchmark binaries (same kernels, self-timing)
#   make test          run the unit tests
#
# CUTLASS (BSD-3) is optional and only needed for `make cutlass`. It is looked
# up in this order: $CUTLASS_ROOT, vendor/cutlass, ../lab-fp4/vendor/cutlass.
# See patches/README.md for the two-line sm_110 enablement patch (CUTLASS 4.4.1).

CUDA_HOME ?= /usr/local/cuda-13.2
NVCC      := $(CUDA_HOME)/bin/nvcc
PY        := .venv/bin/python
ARCH      := -arch=sm_110
ARCHA     := -arch=sm_110a
COMMON    := -O2 -std=c++17 -Xcompiler -fPIC -shared

CUTLASS_DIRS := $(if $(CUTLASS_ROOT),$(CUTLASS_ROOT),$(if $(wildcard vendor/cutlass/include),vendor/cutlass,$(if $(wildcard ../lab-fp4/vendor/cutlass/include),../lab-fp4/vendor/cutlass,)))

.PHONY: venv core cutlass bench-cutlass test clean

venv:
	python3 -m venv .venv
	$(PY) -m pip install torch numpy pytest

build:
	mkdir -p build

build/libthorfp4_core.so: csrc/thor_fp4_core.cu | build
	$(NVCC) $(ARCH) $(COMMON) -o $@ $< -lcublasLt

core: build/libthorfp4_core.so

build/libthorfp4_cutlass.so: csrc/cutlass_kernels.cu | build
	@test -n "$(CUTLASS_DIRS)" || { echo "CUTLASS not found: set CUTLASS_ROOT or run scripts/fetch-cutlass.sh"; exit 1; }
	$(NVCC) $(ARCHA) $(COMMON) --expt-relaxed-constexpr \
	    -I$(CUTLASS_DIRS)/include -I$(CUTLASS_DIRS)/tools/util/include \
	    -o $@ $< -lcublasLt

cutlass: build/libthorfp4_cutlass.so

build/bench_cutlass: csrc/bench_main.cu build/libthorfp4_cutlass.so
	$(NVCC) $(ARCHA) -O2 -std=c++17 -Xlinker -rpath=\$$ORIGIN \
	    -o $@ csrc/bench_main.cu -Lbuild -lthorfp4_cutlass

bench-cutlass: build/bench_cutlass

test: core
	$(PY) -m pytest tests -x -q

clean:
	rm -rf build
