/**
 * @file kernels.hpp
 * @brief Linear-algebra primitives used by the CG solver.
 *
 * This header is the boundary between backend-independent code (the CG
 * outer loop in src/solver.cpp, the driver in src/main.cpp) and the
 * backend-specific implementations. It declares a small set of functions
 * that the CG solver needs, using only standard C++ types -- no CUDA
 * types, no kernel signatures.
 *
 * The functions are implemented twice: once in src/cpu_kernels.cpp using
 * OpenMP, once in src/gpu_kernels.cu using CUDA kernels and cuBLAS. The
 * build system links exactly one of these implementations.
 */

#pragma once

#include "utils.hpp" // DenseMatrix, float

// ---------------------------------------------------------------------------
// Matrix-vector product
// ---------------------------------------------------------------------------

void matvec(const DenseMatrix &A, const float *x, float *y);

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

float dot(const float *a, const float *b, int n);
void axpby(float alpha, const float *x, float beta, float *y, int n);
