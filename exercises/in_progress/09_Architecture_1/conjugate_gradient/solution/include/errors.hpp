#pragma once

// CUDA-only header. Include it only from a .cu file or inside #ifdef USE_GPU.
//
// Both functions are marked inline because this header is included by more than
// one translation unit in the GPU build (solver.cpp and gpu_kernels.cu).
// Without inline, each would emit its own definition and the link would fail
// with "multiple definition".

#include <cuda.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, char const *func, char const *file,
                  int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
inline void check_last(char const *file, int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
