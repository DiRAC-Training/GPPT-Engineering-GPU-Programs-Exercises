#include "idx.hpp"
#include "kernels.hpp"
#include "precision.hpp"

// Dense matrix-vector product: y = A * x.
void matvec(real *y, const real *A, const real *x, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    real sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[idx(i, j, n)] * x[j];
    y[i] = sum;
  }
}

// AXPBY operation: y = alpha * x + beta * y.
void axpby(real *y, const real *x, const real alpha, const real beta,
           const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i] + beta * y[i];
}

// Dot product: result = sum(a[i] * b[i]).
real dot(const real *a, const real *b, const int n) {
  real sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n; i++)
    sum += a[i] * b[i];
  return sum;
}
