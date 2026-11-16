# CMakeLists.txt for the dense FEM heat solver, supporting CPU and GPU builds.
#
# Usage:
#   cmake -B build                 # CPU build (OpenMP)
#   cmake -B build -DUSE_GPU=ON    # GPU build (CUDA + cuBLAS)
#   cmake --build build
#
# The CPU build uses OpenMP. On macOS, this requires Homebrew LLVM (Apple
# clang does not ship -fopenmp). Set CMAKE_CXX_COMPILER to the LLVM clang++
# at configure time, or use a system compiler that supports OpenMP.

cmake_minimum_required(VERSION 3.18)
project(fem_heat LANGUAGES CXX)

option(USE_GPU "Build with GPU support (CUDA + cuBLAS)" OFF)

# ---------------------------------------------------------------------------
# Sources common to both builds
# ---------------------------------------------------------------------------

add_executable(fem_heat
    src/main.cpp
    src/solver.cpp
    src/utils.cpp
    src/test.cpp)

target_include_directories(fem_heat PRIVATE include)
target_compile_features(fem_heat PRIVATE cxx_std_17)

# ---------------------------------------------------------------------------
# CPU build (default)
# ---------------------------------------------------------------------------

find_package(OpenMP REQUIRED)
target_link_libraries(fem_heat PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# GPU build (selected with -DUSE_GPU=ON)
# ---------------------------------------------------------------------------

if(USE_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    find_package(CUDAToolkit REQUIRED)

    target_sources(fem_heat PRIVATE src/gpu_kernels.cu)
    target_compile_definitions(fem_heat PRIVATE USE_GPU)
    target_link_libraries(fem_heat PRIVATE CUDA::cublas)
else()
    target_sources(fem_heat PRIVATE src/cpu_kernels.cpp)
endif()
