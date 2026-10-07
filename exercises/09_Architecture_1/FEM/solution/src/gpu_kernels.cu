/**
 * @file gpu_kernels.cu
 * @brief GPU implementations of the linear-algebra primitives declared in
 *        kernels.hpp.
 *
 * Naive port: each operation launches a separate kernel. The dot product
 * is delegated to cuBLAS (cublasSdot). All vectors are expected to be in
 * managed memory so no explicit data transfers are required at the call
 * site.
 */

#include "kernels.hpp"

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
// Host-side launchers
// ---------------------------------------------------------------------------

static const int BLOCK_SIZE = 256;

void matvec(const DenseMatrix &A, const float *x, float *y) {
    int n    = A.size;
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

float dot(const float *a, const float *b, const int n) {
    float result = 0.0f;
    cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
    return result;
}
