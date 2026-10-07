#include "matmul.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

/*
 * Two approaches are shown below:
 *
 * 1. matmul_cublas_transpose() - Uses CUBLAS_OP_T and transposes output
 *    Adapts the library to match the data layout.
 *    Requires an extra transpose operation on C.
 *
 * 2. matmul_cublas_swap() - Swaps A and B operands
 *    Adapts the code to match the library's expectations.
 *    No extra operations needed - more efficient.
 *
 * Both produce correct results. The swap method is faster because it
 * avoids the extra transpose.
 */

/* ========================================================================
 * Approach 1: Using CUBLAS_OP_T and transposing the output
 * ======================================================================== */

/**
 * @brief Matrix multiplication using transpose flags + output transpose.
 *
 * Uses CUBLAS_OP_T to handle row-major inputs correctly, but the output
 * is still column-major. We use cublasDgeam to transpose C in-place.
 */
void matmul_cublas_transpose(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    double *d_A, *d_B, *d_C, *d_C_temp;
    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_temp, m * n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), m * k * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), k * n * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0;
    const double beta = 0.0;

    /* Use CUBLAS_OP_T to transpose inputs from row-major to column-major.
     * Result is correct but stored in column-major order.
     *
     * Leading dimensions for column-major storage:
     * - lda: rows of A as stored (our row-major A has k "rows" when viewed column-major)
     * - ldb: rows of B as stored (our row-major B has n "rows" when viewed column-major)
     * - ldc: rows of C = m (output is m×n column-major)
     */
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                             m, n, k,
                             &alpha,
                             d_A, k,
                             d_B, n,
                             &beta,
                             d_C_temp, m));

    /* Transpose C from column-major to row-major using cublasDgeam.
     * C_row_major = (C_col_major)^T */
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             n, m,
                             &alpha, d_C_temp, m,
                             &beta, d_C_temp, n,  /* dummy, beta=0 */
                             d_C, n));

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_temp));
    CUBLAS_CHECK(cublasDestroy(handle));
}

/* ========================================================================
 * Approach 2: Swapping A and B operands (more efficient)
 * ======================================================================== */

/**
 * @brief Matrix multiplication using the swap trick.
 *
 * cuBLAS expects column-major layout and outputs in column-major.
 * Our matrices are row-major.
 *
 * Key insight: A row-major matrix, when read as column-major, is its transpose.
 * So cuBLAS sees our A as A^T and our B as B^T.
 *
 * If we swap the order and compute B * A (with OP_N), cuBLAS computes:
 *   B^T * A^T = (A * B)^T
 *
 * The result is stored column-major. When we read it as row-major,
 * we get ((A * B)^T)^T = A * B. Exactly what we want!
 */
void matmul_cublas_swap(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    double *d_A, *d_B, *d_C;
    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), m * k * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), k * n * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0;
    const double beta = 0.0;

    /* Swap A and B, swap m and n.
     * We compute B × A = (n×k) × (k×m) = n×m in cuBLAS's column-major view.
     *
     * Leading dimensions for column-major storage:
     * - lda (for B): rows of B as stored = n (our row-major B viewed column-major)
     * - ldb (for A): rows of A as stored = k (our row-major A viewed column-major)
     * - ldc: rows of result = n (n×m output in column-major)
     *
     * When read as row-major, the n×m column-major result appears as m×n.
     */
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             d_B, n,
                             d_A, k,
                             &beta,
                             d_C, n));

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
}

/* ========================================================================
 * Active implementation
 * ======================================================================== */

void matmul_cublas(const Matrix& A, const Matrix& B, Matrix& C) {
    matmul_cublas_transpose(A, B, C);
    // matmul_cublas_swap(A, B, C);
}
