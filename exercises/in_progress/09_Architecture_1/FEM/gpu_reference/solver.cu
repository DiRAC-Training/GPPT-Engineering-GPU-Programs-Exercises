/**
 * @file solver.cu
 * @brief GPU-offloaded linear algebra routines and CG solver (dense)
 *        using CUDA with managed memory.
 *
 * Naive port: each operation launches a separate kernel. The CG loop
 * runs on the host. Managed memory (cudaMallocManaged) avoids explicit
 * data transfers between kernels.
 *
 * The dot product is delegated to cuBLAS (cublasSdot). Writing an
 * efficient parallel reduction by hand is covered in the algorithms
 * module; interfacing with vendor libraries is the topic of the libraries
 * module. */

#include "solver.hpp"

#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// cuBLAS handle
// ---------------------------------------------------------------------------

// Wrapper struct providing a cuBLAS handle with automatic setup and teardown.
struct CublasHandle {
    cublasHandle_t handle;
    CublasHandle()  { cublasCreate(&handle); }
    ~CublasHandle() { cublasDestroy(handle); }
};
static CublasHandle cublas;

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

/**
 * @brief Dense matrix-vector product kernel: y = A * x.
 *        One thread per row.
 */
__global__ void matvec_kernel(const float *vals, const float *x, float *y,
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += vals[i * n + j] * x[j];
        y[i] = sum;
    }
}

/**
 * @brief AXPBY kernel: y = alpha * x + beta * y.
 *        One thread per element.
 */
__global__ void axpby_kernel(float alpha, const float *x, float beta, float *y,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = alpha * x[i] + beta * y[i];
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

static const int BLOCK_SIZE = 256;

void matvec(const DenseMatrix &A, const float *x, float *y) {
    int n = A.size;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matvec_kernel<<<grid, BLOCK_SIZE>>>(A.vals, x, y, n);
    cudaDeviceSynchronize();
}

void axpby(const float alpha, const float *x, const float beta, float *y,
           const int n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    axpby_kernel<<<grid, BLOCK_SIZE>>>(alpha, x, beta, y, n);
    cudaDeviceSynchronize();
}

// Provided function - no need to port
float dot(const float *a, const float *b, const int n) {
    float result = 0.0f;
    cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
    return result;
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver
// ---------------------------------------------------------------------------

/**
 * @brief Solve A*x = b using the unpreconditioned conjugate gradient method.
 *
 * The CG iteration runs on the host. Each linear algebra operation is
 * a CUDA kernel launch. All vectors use managed memory so no explicit
 * data movement is required.
 *
 * @return Number of iterations performed.
 */
int cg_solve(const DenseMatrix &A, const float *b, float *x, const int max_iter,
             const float tol) {
    int n = A.size;

    float *r, *p, *A_times_p;
    cudaMallocManaged(&r, n * sizeof(float));
    cudaMallocManaged(&p, n * sizeof(float));
    cudaMallocManaged(&A_times_p, n * sizeof(float));

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

    cudaFree(r);
    cudaFree(p);
    cudaFree(A_times_p);

    return iter;
}
