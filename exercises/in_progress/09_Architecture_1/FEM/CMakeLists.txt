# CMakeLists.txt for the dense FEM heat solver, supporting CPU and GPU builds.
#
# Usage:
#   cmake -B build                 # CPU build (OpenMP) -- works out of the box
#   cmake -B build -DUSE_GPU=ON    # GPU build -- requires the GPU branch
#                                  # below to be completed (Task 2).
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

# TODO (Task 1c): once you have created src/cpu_kernels.cpp (Task 1b), add it 
#                 to the CPU build inside the else() branch of the USE_GPU 
#                 block below.

# ---------------------------------------------------------------------------
# GPU build (selected with -DUSE_GPU=ON)
# ---------------------------------------------------------------------------
#
# TODO (Task 2c): inside the `if(USE_GPU)` block below, do the following:
#   - Enable CUDA as a build language (enable_language(CUDA)) and set the
#     CUDA architecture, e.g. set(CMAKE_CUDA_ARCHITECTURES 70). See
#     https://developer.nvidia.com/cuda-gpus for the right value for your
#     GPU.
#   - find_package(CUDAToolkit REQUIRED) so the cuBLAS imported target
#     becomes available.
#   - target_sources(fem_heat PRIVATE src/gpu_kernels.cu) to add the GPU
#     implementation. CMake routes .cu files through nvcc automatically
#     once CUDA is enabled.
#   - target_compile_definitions(fem_heat PRIVATE USE_GPU) so the
#     #ifdef USE_GPU guards in the source code activate.
#   - target_link_libraries(fem_heat PRIVATE CUDA::cublas) for the cuBLAS
#     dot-product call.
#
# In the `else()` branch, add the CPU primitives (same as the Task 1c
# TODO above -- pick one of the two places to add them):
#   target_sources(fem_heat PRIVATE src/cpu_kernels.cpp)
#
# See the [Build system configuration](
#   ../../../docs/60_Architecture_1.md#build-system-configuration) section
# of module 60 for the patterns used here.

if(USE_GPU)
    # TODO: complete the GPU build configuration here.
else()
    # TODO (Task 1c): add src/cpu_kernels.cpp here.
endif()
