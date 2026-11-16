/**
 * @file solver.cpp
 * @brief Backend-independent conjugate gradient solver.
 *
 * The CG outer loop is identical for the CPU and GPU builds: it calls the
 * linear-algebra primitives declared in kernels.hpp without knowing which
 * implementation is linked in. The only build-dependent piece is the
 * allocation and freeing of the temporary vectors r, p and A_times_p,
 * which use the same #ifdef USE_GPU pattern as include/utils.hpp.
 */

#include "solver.hpp"
#include "kernels.hpp"

#include <cmath>
#include <cstring>

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

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
    int n = A.size;

    float *r, *p, *A_times_p;
#ifdef USE_GPU
    cudaMallocManaged(&r, n * sizeof(float));
    cudaMallocManaged(&p, n * sizeof(float));
    cudaMallocManaged(&A_times_p, n * sizeof(float));
#else
    r         = new float[n];
    p         = new float[n];
    A_times_p = new float[n];
#endif

    // Step 1: r_0 = f - K*x_0
    matvec(A, x, r);
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

#ifdef USE_GPU
    cudaFree(r);
    cudaFree(p);
    cudaFree(A_times_p);
#else
    delete[] r;
    delete[] p;
    delete[] A_times_p;
#endif

    return iter;
}
