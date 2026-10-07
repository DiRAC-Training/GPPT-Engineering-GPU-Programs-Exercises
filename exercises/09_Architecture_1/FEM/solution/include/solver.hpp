/**
 * @file solver.hpp
 * @brief Conjugate gradient solver for the dense FEM heat solver.
 *
 * The CG outer loop is implemented in src/solver.cpp and is the same code
 * for both CPU and GPU builds. It calls the linear-algebra primitives
 * declared in kernels.hpp, which are provided in two implementations
 * (src/cpu_kernels.cpp and src/gpu_kernels.cu); the build system selects
 * which one is compiled in.
 */

#pragma once

#include "utils.hpp" // DenseMatrix, float

int cg_solve(const DenseMatrix &A, const float *b, float *x, int max_iter,
             float tol);
