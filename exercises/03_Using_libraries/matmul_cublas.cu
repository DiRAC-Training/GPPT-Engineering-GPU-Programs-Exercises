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
 * @brief Matrix multiplication C = A * B using cuBLAS.
 *
 * Matrix data is stored in column-major order, matching cuBLAS's native layout.
 * Use cublasDgemm. The leading dimension of a column-major matrix is its
 * number of rows.
 *
 * You will need to:
 *   - Create a cuBLAS handle
 *   - Allocate device memory
 *   - Transfer data to the device
 *   - Call cublasDgemm
 *   - Transfer the result back
 *   - Free device memory and destroy the handle
 */
void matmul_cublas(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    // TODO: Implement using cuBLAS
    (void)m; (void)n; (void)k;  // suppress unused warnings
}
