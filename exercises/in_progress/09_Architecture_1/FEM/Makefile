# Makefile for the dense FEM heat solver, supporting CPU and GPU builds.
#
# Usage:
#   make                  # CPU build (OpenMP) -- works out of the box
#   make USE_GPU=1        # GPU build (CUDA + cuBLAS) -- requires the GPU
#                         # branch below to be completed (Task 2).
#
# CPU build details:
#   - macOS: install LLVM via Homebrew for OpenMP support, since Apple clang
#     does not ship with -fopenmp. Set LLVM_DIR if it is not at the default
#     location below.
#   - Linux: replace LLVM_DIR/CXX/CXXFLAGS/LDFLAGS with your system compiler.
#
# GPU build details:
#   - Adjust SM_ARCH to match the target GPU (e.g. sm_70, sm_80, sm_90).
#   - If nvcc and cuBLAS are not on the default search paths, set CUDA_HOME
#     and MATH_LIBS_HOME to the relevant directories (e.g. for the NVIDIA HPC
#     SDK, $(NVHPC_ROOT)/cuda and $(NVHPC_ROOT)/math_libs).

LLVM_DIR ?= /opt/homebrew/opt/llvm
CXX       = $(LLVM_DIR)/bin/clang++
NVCC      = nvcc
SM_ARCH   = sm_70

SRC_DIR     = src
INCLUDE_DIR = include
BUILD_DIR   = build

# ---------------------------------------------------------------------------
# CPU build (default)
# ---------------------------------------------------------------------------

CXXFLAGS    = -std=c++17 -O3 -fopenmp -march=native -DNDEBUG -I$(INCLUDE_DIR) \
              -I$(LLVM_DIR)/include
LDFLAGS     = -fopenmp -L$(LLVM_DIR)/lib

# TODO (Task 1c): once you have created src/cpu_kernels.cpp (Task 1b), add
#                 $(BUILD_DIR)/cpu_kernels.o to KERNELS_OBJ below so the
#                 build picks it up. The compile rule for it is already
#                 defined further down in this file.
KERNELS_OBJ =

# ---------------------------------------------------------------------------
# GPU build (selected with USE_GPU=1)
# ---------------------------------------------------------------------------
#
# TODO (Task 2c): inside the `ifdef USE_GPU` block below, override CXX,
#                 CXXFLAGS, LDFLAGS and KERNELS_OBJ so that:
#   - CXX uses the CUDA compiler ($(NVCC)) rather than the host compiler.
#     nvcc accepts plain C++ for the .cpp files and is the only compiler
#     that can build src/gpu_kernels.cu.
#   - CXXFLAGS sets the CUDA architecture (-arch=$(SM_ARCH)), passes
#     -DUSE_GPU so the source-level guards activate, and points at the
#     project includes. Drop the OpenMP flags that the CPU build needs.
#   - LDFLAGS links cuBLAS (-lcublas).
#   - KERNELS_OBJ becomes $(BUILD_DIR)/gpu_kernels.o so the build picks
#     up the GPU implementation rather than the CPU one.
#
# See the [Build system configuration](
#   ../../../docs/60_Architecture_1.md#build-system-configuration) section
# of module 60 for the patterns used here.

ifdef USE_GPU
  # TODO: complete the GPU build configuration here.
endif

COMMON_OBJS = $(BUILD_DIR)/main.o $(BUILD_DIR)/solver.o \
              $(BUILD_DIR)/utils.o $(BUILD_DIR)/test.o

# ---------------------------------------------------------------------------
# Build rules
# ---------------------------------------------------------------------------

.PHONY: all clean

all: $(BUILD_DIR)/fem_heat

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/test.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/solver.o: $(SRC_DIR)/solver.cpp $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/kernels.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test.o: $(SRC_DIR)/test.cpp $(INCLUDE_DIR)/test.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/cpu_kernels.o: $(SRC_DIR)/cpu_kernels.cpp $(INCLUDE_DIR)/kernels.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/gpu_kernels.o: $(SRC_DIR)/gpu_kernels.cu $(INCLUDE_DIR)/kernels.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/fem_heat: $(COMMON_OBJS) $(KERNELS_OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD_DIR) temperature.dat
