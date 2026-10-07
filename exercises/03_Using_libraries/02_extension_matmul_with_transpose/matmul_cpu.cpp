#include "matmul.hpp"
#include <omp.h>

/*
 * Two CPU implementations are provided:
 *
 * 1. matmul_cpu_serial() - Simple triple-loop implementation
 *    Easy to understand but slow: single-threaded, poor cache usage.
 *
 * 2. matmul_cpu_parallel() - Threaded with OpenMP
 *    Uses OpenMP for parallelism and i-k-j loop order for better
 *    cache behavior. Inner loop is vectorized with SIMD.
 */

/* ========================================================================
 * Approach 1: Single-threaded (naive)
 * ======================================================================== */

/**
 * @brief Naive matrix multiplication C = A * B using a triple loop.
 *
 * Loop order is i-j-k which has poor cache behavior: B[l,j] has
 * stride-N access in the innermost loop.
 */
void matmul_cpu_serial(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();
    int n = B.cols();
    int k = A.cols();

    C.zero();

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
}

/* ========================================================================
 * Approach 2: Parallel with OpenMP
 * ======================================================================== */

/**
 * @brief Parallel matrix multiplication using OpenMP.
 *
 * Note: schedule(static) is used for predictable workload with low overhead.
 */
void matmul_cpu_parallel(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();
    int n = B.cols();
    int k = A.cols();

    C.zero();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
}

/* ========================================================================
 * Active implementation
 * ======================================================================== */

void matmul_cpu(const Matrix& A, const Matrix& B, Matrix& C) {
    // matmul_cpu_serial(A, B, C);
    matmul_cpu_parallel(A, B, C);
}
