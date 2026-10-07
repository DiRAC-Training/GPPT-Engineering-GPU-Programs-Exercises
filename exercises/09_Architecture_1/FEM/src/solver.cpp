/**
 * @file solver.cpp
 * @brief Implementation of linear algebra routines and CG solver (dense).
 *
 * GPU porting target: all functions in this file operate on DenseMatrix
 * and raw vectors with clean, portable signatures.
 */

#include "solver.hpp"

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Matrix-vector product
// ---------------------------------------------------------------------------

/**
 * @brief Dense matrix-vector product: y = A * x.
 */
void matvec(const DenseMatrix &A, const float *x, float *y) {
#pragma omp parallel for
    for (int i = 0; i < A.size; i++) {
        float sum = 0.0;
        for (int j = 0; j < A.size; j++)
            sum += A(i, j) * x[j];
        y[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

/**
 * @brief Dot product: result = sum(a[i] * b[i]).
 */
float dot(const float *a, const float *b, const int n) {
    float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/**
 * @brief AXPBY operation: y = alpha * x + beta * y.
 */
void axpby(const float alpha, const float *x, const float beta, float *y,
           const int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver
// ---------------------------------------------------------------------------

/**
 * @brief Solve A*x = b using the unpreconditioned conjugate gradient method.
 *
 * @return Number of iterations performed.
 */
int cg_solve(const DenseMatrix &A, const float *b, float *x, const int max_iter,
             const float tol) {
    int n           = A.size;
    float *r         = new float[n];
    float *p         = new float[n];
    float *A_times_p = new float[n];

    // Step 1: r_0 = f - K*x_0
    matvec(A, x, r);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        r[i] = b[i] - r[i];

    // Step 2: p_0 = r_0
    std::memcpy(p, r, n * sizeof(float));

    float residual_sq_old = dot(r, r, n);
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        // Step 3a: alpha_k = (r_k . r_k) / (p_k . K*p_k)
        matvec(A, p, A_times_p);
        float alpha = residual_sq_old / dot(p, A_times_p, n);

        // Step 3b: x_{k+1} = x_k + alpha_k * p_k
        axpby(alpha, p, 1.0, x, n);

        // Step 3c: r_{k+1} = r_k - alpha_k * K*p_k
        axpby(-alpha, A_times_p, 1.0, r, n);

        // Step 3d: convergence check -- stop if ||r_{k+1}|| < tol
        float residual_sq_new = dot(r, r, n);
        if (std::sqrt(residual_sq_new) < tol) {
            iter++;
            break;
        }

        // Step 3e: beta_k = (r_{k+1} . r_{k+1}) / (r_k . r_k)
        //          p_{k+1} = r_{k+1} + beta_k * p_k
        float beta = residual_sq_new / residual_sq_old;
        axpby(1.0, r, beta, p, n);

        residual_sq_old = residual_sq_new;
    }

    delete[] r;
    delete[] p;
    delete[] A_times_p;
    return iter;
}
