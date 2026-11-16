/**
 * @file cpu_kernels.cpp
 * @brief CPU implementations of the linear-algebra primitives declared in
 *        kernels.hpp. Loops are parallelised with OpenMP.
 */

#include "kernels.hpp"

// ---------------------------------------------------------------------------
// Matrix-vector product
// ---------------------------------------------------------------------------

/**
 * @brief Dense matrix-vector product: y = A * x.
 */
void matvec(const DenseMatrix &A, const float *x, float *y) {
#pragma omp parallel for
    for (int i = 0; i < A.size; i++) {
        float sum = 0.0;
        for (int j = 0; j < A.size; j++)
            sum += A(i, j) * x[j];
        y[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Vector operations
// ---------------------------------------------------------------------------

/**
 * @brief Dot product: result = sum(a[i] * b[i]).
 */
float dot(const float *a, const float *b, const int n) {
    float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/**
 * @brief AXPBY operation: y = alpha * x + beta * y.
 */
void axpby(const float alpha, const float *x, const float beta, float *y,
           const int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        y[i] = alpha * x[i] + beta * y[i];
}
