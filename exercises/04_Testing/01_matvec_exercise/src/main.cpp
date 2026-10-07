#include <cuda_runtime.h>

#include "kernels.hpp"

int main() {

  // Size of problem
  const int n = 4096;

  // Declare variables
  float *A, *y, *x;
  cudaMallocManaged(&A, n * n * sizeof(float));
  cudaMallocManaged(&y, n * sizeof(float));
  cudaMallocManaged(&x, n * sizeof(float));

  // Initialise variables
  for (int i = 0; i < n; i++) {
    y[i] = 0;
    x[i] = i;
    for (int j = 0; j < n; j++) {
      int fill = i == j ? 1 : 0;
      A[i * n + j] = fill;
    }
  }

  // Call run_matvec
  run_matvec(y, A, x, n);
}
