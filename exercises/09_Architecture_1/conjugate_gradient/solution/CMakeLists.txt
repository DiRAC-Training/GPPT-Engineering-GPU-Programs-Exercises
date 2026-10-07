# CMakeLists.txt for the conjugate gradient solver, supporting CPU and GPU
# builds.
#
# Usage:
#   cmake -B build                 # CPU build (OpenMP)
#   cmake -B build -DUSE_GPU=ON    # GPU build (CUDA + cuBLAS)
#   cmake --build build

cmake_minimum_required(VERSION 3.24)
project(cg LANGUAGES CXX)

option(USE_GPU "Build with GPU support (CUDA + cuBLAS)" OFF)

# Optimise by default, matching the Makefile. Release supplies -O3 -DNDEBUG;
# the remaining flags are set on the target below.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compile for the GPU in this machine. Set explicitly (e.g. -DARCH=80) when
# building on a node with no GPU attached (e.g. login node).
set(ARCH native CACHE STRING "CUDA architecture to compile for")

# ---------------------------------------------------------------------------
# Sources common to both builds
# ---------------------------------------------------------------------------

add_executable(cg
    src/main.cpp
    src/solver.cpp)

target_include_directories(cg PRIVATE include)

# Restricted to C++ so that nvcc does not receive -march=native when compiling
# the CUDA sources in the GPU build.
target_compile_options(cg PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -march=native>)

# ---------------------------------------------------------------------------
# OpenMP, used by both builds: the GPU build still assembles the system on the
# host.
# ---------------------------------------------------------------------------

find_package(OpenMP REQUIRED)
target_link_libraries(cg PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Backend selection: the CPU and GPU kernels are interchangeable behind
# kernels.hpp, so switching backend only changes which one is compiled.
# ---------------------------------------------------------------------------

if(USE_GPU)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set_target_properties(cg PROPERTIES CUDA_ARCHITECTURES "${ARCH}")
    target_sources(cg PRIVATE src/gpu_kernels.cu)
    target_compile_definitions(cg PRIVATE USE_GPU)
    target_link_libraries(cg PRIVATE CUDA::cublas)
else()
    target_sources(cg PRIVATE src/cpu_kernels.cpp)
endif()
