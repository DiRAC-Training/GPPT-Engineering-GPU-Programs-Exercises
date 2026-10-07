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

/**
 * @brief Matrix multiplication using cublasDgemm.
 *
 * Matrix data is column-major, matching cuBLAS's native layout, so no
 * reordering or transpose tricks are needed. The leading dimension of a
 * column-major matrix is its number of rows.
 */
void matmul_cublas(const Matrix& A, const Matrix& B, Matrix& C) {
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
    const double beta  = 0.0;

    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_A, m,
                             d_B, k,
                             &beta,
                             d_C, m));

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
}
