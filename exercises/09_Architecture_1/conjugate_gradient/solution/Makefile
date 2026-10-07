# Makefile for the conjugate gradient solver, supporting CPU and GPU builds.
#
# Usage:
#   make                  # CPU build (OpenMP)
#   make USE_GPU=1        # GPU build (CUDA + cuBLAS)
#   make fmt              # reformat the sources
#
# Run `make clean` when switching between the CPU and GPU builds. Both produce
# the same object and binary names, so make sees main.o and solver.o as up to
# date and reuses them, even though they were compiled for the other backend.
#
# The host compiler defaults to nvc++, which supports OpenMP and can link the
# CUDA libraries. Override it for a different toolchain, e.g.
#   make CXX=g++
#
# If nvcc and the CUDA headers and libraries are not on the default search
# paths, set CUDA_PATH to the toolkit root, e.g. for the NVIDIA HPC SDK:
#   make USE_GPU=1 CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda

CXX  = nvc++
NVCC = nvcc

# Optional: locate a CUDA toolkit that is not on the default search paths.
CUDA_PATH ?=

# Compile for the GPU in this machine. Set explicitly (e.g. ARCH=sm_80) when
# building on a node with no GPU attached (e.g. login node).
ARCH ?= native

HEADERS = $(wildcard include/*.hpp)

BLOCK_SIZE  ?= 128
EXTRA_FLAGS  = -DBLOCK_SIZE=$(BLOCK_SIZE)

CXXFLAGS  = -Wall -O3 -march=native -fopenmp -DNDEBUG -Iinclude/ -I"$(CUDA_PATH)/include"
NVCCFLAGS = -O3 -arch=$(ARCH) -DNDEBUG -Iinclude/
LDFLAGS   = -fopenmp

# ---------------------------------------------------------------------------
# Backend selection: the CPU and GPU kernels are interchangeable behind
# kernels.hpp, so switching backend only changes which one is compiled in.
# ---------------------------------------------------------------------------

ifdef USE_GPU
  EXTRA_FLAGS += -DUSE_GPU
  OBJS         = build/main.o build/solver.o build/gpu_kernels.o
  LDLIBS       = -L"$(CUDA_PATH)/lib" -lcublas -lcudart
else
  OBJS         = build/main.o build/solver.o build/cpu_kernels.o
  LDLIBS       =
endif

# ---------------------------------------------------------------------------
# Build rules
# ---------------------------------------------------------------------------

.PHONY: all clean fmt

all: build/cg

build:
	mkdir -p build

build/%.o: src/%.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(EXTRA_FLAGS) -c $< -o $@

build/%.o: src/%.cu $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) $(EXTRA_FLAGS) -c $< -o $@

build/cg: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

fmt:
	clang-format --style=llvm -i src/*.cpp src/*.cu $(HEADERS)

clean:
	rm -rf build
