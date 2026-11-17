#include <cuda_runtime.h>

#include "kernels.hpp"
// TODO Exercise 2.3: Uncomment to include the device function declaration.
// #include "kernels.cuh"

/// Device kernel that works out the correct index for access.
__device__ int get_index(int i, int j, int n) { return i * n + j; }

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
  cudaDeviceSynchronize();
}

/// TODO: Exercise 1.2: Write a wrapper for metvec_kernel that carries out all
/// CUDA commands
void custom_matvec_wrapper(float *y, const float *A, const float *x,
                           const int n) {
  // Assign memory accessible to the device

  // Launch kernel

  // Synchronise
  cudaDeviceSynchronize();
}

// TODO: Exercise 2.1: Write a host function that calls `get_index`
void run_get_index(int i, int j, int n, int *result) {}

// TODO: Exercise 2.2: Write a thin global wrapper kernel for the get_index
// function
