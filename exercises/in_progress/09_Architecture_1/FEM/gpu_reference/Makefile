# Makefile for the dense FEM heat solver with CUDA managed memory.
#
# All arrays use cudaMallocManaged so that host assembly and GPU kernels
# can access the same pointers without explicit copies.
#
# Adjust SM_ARCH to match the target GPU (e.g. sm_70, sm_80, sm_90).
#
# If nvcc and cuBLAS are not on the default search paths, provide CUDA_PATH
# as a single root that contains both, e.g. for the NVIDIA HPC SDK:
#   make CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2025
# This resolves to CUDA_HOME=$(CUDA_PATH)/cuda and
# MATH_LIBS_HOME=$(CUDA_PATH)/math_libs. For non-standard layouts, set
# CUDA_HOME and MATH_LIBS_HOME individually instead.

CUDA_PATH ?=
ifneq ($(strip $(CUDA_PATH)),)
CUDA_HOME      ?= $(CUDA_PATH)/cuda
MATH_LIBS_HOME ?= $(CUDA_PATH)/math_libs
endif

CUDA_HOME      ?=
MATH_LIBS_HOME ?=
SM_ARCH = sm_70

ifeq ($(strip $(CUDA_HOME)),)
NVCC = nvcc
CUDA_INC =
CUDA_LIB =
else
NVCC = $(CUDA_HOME)/bin/nvcc
CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64
endif

ifeq ($(strip $(MATH_LIBS_HOME)),)
MATH_INC =
MATH_LIB =
else
MATH_INC = -I$(MATH_LIBS_HOME)/include
MATH_LIB = -L$(MATH_LIBS_HOME)/lib64 -Xlinker -rpath -Xlinker $(MATH_LIBS_HOME)/lib64
endif

NVCCFLAGS = -std=c++17 -O3 -arch=$(SM_ARCH) -DNDEBUG -I$(INCLUDE_DIR) $(CUDA_INC) $(MATH_INC)
LDFLAGS = $(CUDA_LIB) $(MATH_LIB) -lcublas

SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

.PHONY: all clean

all: $(BUILD_DIR)/fem_heat

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/solver.o: $(SRC_DIR)/solver.cu $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/test.o: $(SRC_DIR)/test.cpp $(INCLUDE_DIR)/test.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INCLUDE_DIR)/solver.hpp $(INCLUDE_DIR)/test.hpp $(INCLUDE_DIR)/utils.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/fem_heat: $(BUILD_DIR)/solver.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/test.o $(BUILD_DIR)/main.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
