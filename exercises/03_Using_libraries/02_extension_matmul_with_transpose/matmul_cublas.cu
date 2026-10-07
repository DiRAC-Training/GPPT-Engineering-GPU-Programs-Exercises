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

// TODO: Add necessary includes

/**
 * @brief Matrix multiplication C = A * B using cuBLAS.
 *
 * Consult the cuBLAS documentation for cublasDgemm.
 * Consider how the library expects matrix data to be laid out.
 *
 * You will need to:
 *   - Allocate device memory
 *   - Transfer data to the device
 *   - Perform the computation
 *   - Transfer the result back
 *   - Free device memory
 */
void matmul_cublas(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    // TODO: Implement using cuBLAS
    (void)m; (void)n; (void)k;  // suppress unused warnings
}
