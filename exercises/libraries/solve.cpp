#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "matmul.hpp"

/* Default square size */
constexpr int DEFAULT_M = 512;   // rows of A and C
constexpr int DEFAULT_N = 512;   // cols of B and C
constexpr int DEFAULT_K = 512;   // cols of A, rows of B

/**
 * @brief Returns the current wall-clock time in seconds.
 */
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char* argv[]) {
    int M = DEFAULT_M;
    int N = DEFAULT_N;
    int K = DEFAULT_K;

    if (argc == 2) {
        // Single arg: square matrices
        int size = std::atoi(argv[1]);
        if (size > 0) M = N = K = size;
    } else if (argc >= 4) {
        // Three args: M K N (A is MxK, B is KxN, C is MxN)
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    printf("GEMM: C(%d x %d) = A(%d x %d) * B(%d x %d)\n\n", M, N, M, K, K, N);

    Matrix A(M, K);
    Matrix B(K, N);
    Matrix C_cpu(M, N);
    Matrix C_blas(M, N);
    Matrix C_cublas(M, N);

    A.randomize(42);
    B.randomize(43);

    double t0, t1;

    // CPU implementation (baseline)
    t0 = get_time_sec();
    matmul_cpu(A, B, C_cpu);
    t1 = get_time_sec();

    double elapsed_cpu = t1 - t0;
    double gflops_cpu = (2.0 * M * N * K) / (elapsed_cpu * 1e9);

    printf("CPU (OpenMP):\n");
    printf("  Time: %.4f seconds\n", elapsed_cpu);
    printf("  Performance: %.2f GFLOPS\n\n", gflops_cpu);

    // TODO: Uncomment when matmul_blas is implemented
    // t0 = get_time_sec();
    // matmul_blas(A, B, C_blas);
    // t1 = get_time_sec();
    //
    // double elapsed_blas = t1 - t0;
    // double gflops_blas = (2.0 * M * N * K) / (elapsed_blas * 1e9);
    //
    // printf("BLAS:\n");
    // printf("  Time: %.4f seconds\n", elapsed_blas);
    // printf("  Performance: %.2f GFLOPS\n", gflops_blas);
    // printf("  Verification: %s\n", (C_cpu == C_blas) ? "PASSED" : "FAILED");
    // printf("  Speedup: %.2fx\n\n", elapsed_cpu / elapsed_blas);

    // TODO: Uncomment when matmul_cublas is implemented
    // t0 = get_time_sec();
    // matmul_cublas(A, B, C_cublas);
    // t1 = get_time_sec();
    //
    // double elapsed_cublas = t1 - t0;
    // double gflops_cublas = (2.0 * M * N * K) / (elapsed_cublas * 1e9);
    //
    // printf("cuBLAS:\n");
    // printf("  Time: %.4f seconds\n", elapsed_cublas);
    // printf("  Performance: %.2f GFLOPS\n", gflops_cublas);
    // printf("  Verification: %s\n", (C_cpu == C_cublas) ? "PASSED" : "FAILED");
    // printf("  Speedup: %.2fx\n\n", elapsed_cpu / elapsed_cublas);

    // Suppress unused variable warnings
    (void)C_blas;
    (void)C_cublas;

    return 0;
}
