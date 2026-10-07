#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "errors.hpp"
#include "kernels.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

// Row-major index into an n x n matrix, callable from device code. The host
// backend has its own idx() in idx.hpp; the two cannot share one definition
// because a function cannot be overloaded on execution space alone.
__device__ inline int idx(int i, int j, int n) { return i * n + j; }

// Wrapper struct providing a cuBLAS handle with automatic setup and teardown.
struct CublasHandle {
  cublasHandle_t handle;
  CublasHandle() {
    cublasCreate(&handle);
    cublasSetStream(handle, 0); // share the default stream with our kernels
  }
  ~CublasHandle() { cublasDestroy(handle); }
};
static CublasHandle cublas;

// Dense matrix-vector product kernel: y = A * x. One thread per row.
__global__ void matvec_kernel(real *y, const real *A, const real *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    real sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[idx(i, j, n)] * x[j];
    y[i] = sum;
  }
}

// AXPBY kernel: y = alpha * x + beta * y. One thread per element.
__global__ void axpby_kernel(real *y, const real *x, real alpha, real beta,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = alpha * x[i] + beta * y[i];
}

void matvec(real *y, const real *A, const real *x, const int n) {
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; // one thread per row
  matvec_kernel<<<grid, BLOCK_SIZE>>>(y, A, x, n);
  CHECK_LAST_CUDA_ERROR();
}

void axpby(real *y, const real *x, const real alpha, const real beta,
           const int n) {
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  axpby_kernel<<<grid, BLOCK_SIZE>>>(y, x, alpha, beta, n);
  CHECK_LAST_CUDA_ERROR();
}

// Dot product: result = sum(a[i] * b[i]).
real dot(const real *a, const real *b, const int n) {
  real result = 0.0f;
#ifdef SINGLE_PRECISION
  cublasSdot(cublas.handle, n, a, 1, b, 1, &result);
#else
  cublasDdot(cublas.handle, n, a, 1, b, 1, &result);
#endif
  CHECK_LAST_CUDA_ERROR();
  return result;
}
