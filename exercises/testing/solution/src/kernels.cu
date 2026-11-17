#include <algorithm>
#include <cuda_runtime.h>
#include <iterator>

#include "kernels.cuh"
#include "kernels.hpp"

/// Device kernel that works out the correct index for access.
__device__ __host__ int get_index(int i, int j, int n) { return i * n + j; }

/// Dense matrix-vector product kernel: y = A * x. One thread per row.
__global__ void matvec_kernel(float *y, const float *vals, const float *x,
                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += vals[get_index(i, j, n)] * x[j];
    y[i] = sum;
  }
}

static const int BLOCK_SIZE = 256;

/// Launches the matvec kernel
void run_matvec(float *y, const float *A, const float *x, const int n) {
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  matvec_kernel<<<grid, BLOCK_SIZE>>>(y, A, x, n);
}

/// TODO: Exercise 1.2: Write a wrapper for metvec_kernel that carries out all
/// CUDA commands
void custom_matvec_wrapper(float *y_h, const float *A_h, const float *x_h,
                           const int n) {
  // Allocate device memory
  float *A_d, *y_d, *x_d;
  cudaMalloc(&A_d, n * n * sizeof(float));
  cudaMalloc(&y_d, n * sizeof(float));
  cudaMalloc(&x_d, n * sizeof(float));

  // Copy inputs to the device
  cudaMemcpy(A_d, A_h, n * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel
  run_matvec(y_d, A_d, x_d, n);

  // Copy the result to the host
  cudaMemcpy(y_h, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(A_d);
  cudaFree(y_d);
  cudaFree(x_d);
}

// TODO: Exercise 2.1: Write a host function that calls `get_index`
void run_get_index(int i, int j, int n, int *result) {
  *result = get_index(i, j, n);
}
