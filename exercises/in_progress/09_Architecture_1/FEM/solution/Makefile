# Makefile for the dense FEM heat solver, supporting CPU and GPU builds.
#
# Usage:
#   make                  # CPU build (OpenMP)
#   make USE_GPU=1        # GPU build (CUDA + cuBLAS)
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
KERNELS_OBJ = $(BUILD_DIR)/cpu_kernels.o

# ---------------------------------------------------------------------------
# GPU build (selected with USE_GPU=1)
# ---------------------------------------------------------------------------

ifdef USE_GPU
  CXX         = $(NVCC)
  CXXFLAGS    = -std=c++17 -O3 -arch=$(SM_ARCH) -DUSE_GPU -DNDEBUG -I$(INCLUDE_DIR)
  LDFLAGS     = -lcublas
  KERNELS_OBJ = $(BUILD_DIR)/gpu_kernels.o
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
