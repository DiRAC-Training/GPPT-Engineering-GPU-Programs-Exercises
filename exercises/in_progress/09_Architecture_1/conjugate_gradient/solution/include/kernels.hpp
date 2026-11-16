#pragma once

#include "precision.hpp"

// Linear-algebra primitives used by cg_solve. This header is the boundary
// between the backend-independent solver (src/solver.cpp) and the two
// interchangeable implementations: src/cpu_kernels.cpp (OpenMP) and
// src/gpu_kernels.cu (CUDA + cuBLAS). The build system links exactly one.
//
// Declared with standard C++ types only -- no CUDA types, no kernel
// signatures, no CUDA includes. Anything CUDA-specific here would force every
// file that includes this header through nvcc and break the CPU-only build.
//
// Note the deliberate absence of #include "idx.hpp": the GPU backend defines
// its own __device__ idx(), which would clash with a host idx() pulled in here
// (functions cannot be overloaded on execution space alone).

void matvec(real *y, const real *A, const real *x, const int n);
real dot(const real *a, const real *b, const int n);
void axpby(real *y, const real *x, real alpha, real beta, int n);
