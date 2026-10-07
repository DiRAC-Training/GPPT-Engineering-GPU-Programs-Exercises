#include "matmul.hpp"
#include <cblas.h>

/**
 * @brief Matrix multiplication C = A * B using BLAS.
 *
 * Matrix data is stored in column-major order, matching BLAS's native layout.
 * Use cblas_dgemm. The leading dimension of a column-major matrix is its
 * number of rows.
 */
void matmul_blas(const Matrix& A, const Matrix& B, Matrix& C) {
    int m = A.rows();   // rows of A and C
    int n = B.cols();   // cols of B and C
    int k = A.cols();   // cols of A, rows of B

    // TODO: Implement using cblas_dgemm
    (void)m; (void)n; (void)k;  // suppress unused warnings
}
