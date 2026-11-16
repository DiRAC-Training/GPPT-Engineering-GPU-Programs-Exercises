/**
 * @file kernels.hpp
 * @brief Linear-algebra primitives used by the CG solver.
 *
 * This header is the planned boundary between the backend-independent code
 * (the CG outer loop and the driver) and the backend-specific
 * implementations. It should declare a small set of functions that the CG
 * solver needs, using only standard C++ types -- no CUDA types, no kernel
 * signatures. The same declarations will be satisfied by both
 * src/cpu_kernels.cpp (OpenMP) and src/gpu_kernels.cu (CUDA + cuBLAS),
 * and the build system picks which one to compile.
 *
 * TODO (Task 1a): move the matvec, dot and axpby declarations here from
 *                 include/solver.hpp, leaving only cg_solve in solver.hpp.
 *                 The declarations to move are the three that currently
 *                 sit under the "Matrix-vector product" and
 *                 "Vector operations" banners in solver.hpp.
 */

#pragma once

#include "utils.hpp" // DenseMatrix, float

// TODO (Task 1a): declarations go here.
