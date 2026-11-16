/**
 * @file solver.hpp
 * @brief Linear algebra routines and CG solver for the dense FEM heat solver.
 *
 * This file is the primary GPU porting target. It contains the matrix-vector
 * product, dot product, AXPBY, and conjugate gradient solver -- all operating
 * on DenseMatrix and raw vectors with no dependency on Parameters.
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

// ---------------------------------------------------------------------------
// CG solver
// ---------------------------------------------------------------------------

int cg_solve(const DenseMatrix &A, const float *b, float *x, int max_iter,
             float tol);
