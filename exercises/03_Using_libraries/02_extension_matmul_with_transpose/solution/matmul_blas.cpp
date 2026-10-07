#include "matmul.hpp"
#include <cblas.h>

/*
 * Two approaches are shown below:
 *
 * 1. matmul_blas_cblas() - Uses cblas_dgemm with CblasRowMajor
 *    Adapts the library to match the data layout.
 *    CBLAS provides native row-major support.
 *
 * 2. matmul_blas_fortran() - Swaps A and B operands
 *    Adapts the code to match the library's expectations.
 *    Same swap trick as cuBLAS.
 *
 * Both produce correct results. The cblas approach is simpler for row-major
 * data, which the fortran blas approach is more similar to cublas.
 */

/* ========================================================================
 * Approach 1: Using cblas_dgemm with row-major support
 * ======================================================================== */

/**
 * @brief Matrix multiplication using cblas_dgemm with CblasRowMajor.
 *
 * CBLAS provides native row-major support via the first parameter.
 * This is the simplest approach for C/C++ code.
 */
void matmul_blas_cblas(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    /* Leading dimensions for row-major storage:
     * - lda: cols of A = k (stride between rows)
     * - ldb: cols of B = n
     * - ldc: cols of C = n
     */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0, A.data(), k,
                     B.data(), n,
                0.0, C.data(), n);
}

/* ========================================================================
 * Approach 2: Using Fortran dgemm with swap trick
 * ======================================================================== */

extern "C" {
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb,
                const double* beta, double* C, const int* ldc);
}

/**
 * @brief Matrix multiplication using the swap trick.
 *
 * Fortran BLAS expects column-major layout and outputs in column-major.
 * Our matrices are row-major.
 *
 * A row-major matrix, when read as column-major, is its transpose.
 * So BLAS sees A as A^T and B as B^T.
 *
 * If we swap the order and compute B * A (with 'N'), BLAS computes:
 *   B^T * A^T = (A * B)^T
 *
 * The result is stored column-major. When we read it as row-major,
 * we get ((A * B)^T)^T = A * B. Exactly what we want!
 */
void matmul_blas_fortran(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    double alpha = 1.0;
    double beta = 0.0;
    char no_trans = 'N';

    /* Swap A and B, swap m and n.
     * We compute B × A = (n×k) × (k×m) = n×m in BLAS's column-major view.
     *
     * Leading dimensions for column-major storage:
     * - lda (for B): rows of B as stored = n (our row-major B viewed column-major)
     * - ldb (for A): rows of A as stored = k (our row-major A viewed column-major)
     * - ldc: rows of result = n (n×m output in column-major)
     *
     * When read as row-major, the n×m column-major result appears as m×n.
     */
    dgemm_(&no_trans, &no_trans, &n, &m, &k,
           &alpha, B.data(), &n,
                   A.data(), &k,
           &beta,  C.data(), &n);
}

/* ========================================================================
 * Active implementation
 * ======================================================================== */

void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C) {
    matmul_blas_cblas(A, B, C);
    // matmul_blas_fortran(A, B, C);
}
