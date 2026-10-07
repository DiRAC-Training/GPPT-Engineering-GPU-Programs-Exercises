cmake_minimum_required(VERSION 3.18)
project(gemm_solution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")

find_package(OpenMP REQUIRED)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(CBLAS_INCLUDE_DIR cblas.h
    HINTS
        /usr/include
        /usr/local/include
        /usr/include/openblas
        /usr/local/opt/openblas/include
        $ENV{MKLROOT}/include
)

add_executable(solution
    solve.cpp
    ../matmul_cpu.cpp
    matmul_blas.cpp
    matmul_cublas.cu
)

target_include_directories(solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(solution
    ${BLAS_LIBRARIES}
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

if(CBLAS_INCLUDE_DIR)
    target_include_directories(solution PRIVATE ${CBLAS_INCLUDE_DIR})
endif()
