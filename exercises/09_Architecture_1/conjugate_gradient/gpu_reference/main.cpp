#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "idx.hpp"
#include "precision.hpp"
#include "solver.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

const bool RANDOMISE_SEED = true;

// Diagonal shift that makes the matrix symmetric positive definite. DIAG_SCALE
// > 0 uses f*sqrt(n), giving a poorly conditioned (but reliably solvable)
// system that takes more CG iterations to converge; otherwise n/32, which is
// better conditioned and converges in fewer.
const real DIAG_SCALE = 0.42;

static std::mt19937 rng;

/// Create a rng
void init_rng(bool randomise) {
  std::mt19937::result_type seed = 42;
  if (randomise) {
    seed = static_cast<std::mt19937::result_type>(
        std::chrono::system_clock::now().time_since_epoch().count());
  }
  rng.seed(seed);
}

/// Generate b from Ax = b
void calc_b(real *b, const real *A, const real *x, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    b[i] = 0.0;

    for (int j = 0; j < n; ++j) {
      b[i] += A[idx(i, j, n)] * x[j];
    }
  }
}

/// Fill x with random values in [min, max)
void fill_rand_vec(real *x, int size, real min, real max) {
  std::uniform_real_distribution<real> dist(min, max);
  for (int i = 0; i < size; ++i) {
    x[i] = dist(rng);
  }
}

/// Create a random, positive-definite matrix
void generate_positive_definite(real *A, int n) {
  std::cout << "Generating matrix\n";
  // Create a totally random matrix
  auto B = std::vector<real>(n * n);
  fill_rand_vec(B.data(), n * n, 0.0, 1.0);

// Make a symmetric matrix
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[idx(i, j, n)] = (B[idx(i, j, n)] + B[idx(j, i, n)]) / 2.0;
    }
  }

  // DIAG_SCALE > 0 uses f*sqrt(n) (barely SPD, ill-conditioned); otherwise
  // n/32.
  real diag_shift = DIAG_SCALE > 0 ? DIAG_SCALE * std::sqrt(real(n))
                                   : fmax(real(n) / 32, 1.0);
  for (int i = 0; i < n; ++i)
    A[idx(i, i, n)] += diag_shift;
}

int main() {
  const int n = 8192;
  const int cg_max_iter = 128;

  real *A = nullptr;
  cudaMallocManaged(&A, (size_t)n * n * sizeof(real));
  real *b = nullptr;
  cudaMallocManaged(&b, n * sizeof(real));
  real *x = nullptr;
  cudaMallocManaged(&x, n * sizeof(real));

  // This doesn't ever need to go on the GPU
  real *x_soln = new real[n];

  init_rng(RANDOMISE_SEED);

  // Initial conditions
  generate_positive_definite(A, n); // Generate positive def matrix
  std::cout << "Generating random solution\n";
  fill_rand_vec(x_soln, n, -1.0, 1.0); // Generate a random solution vector
  std::cout << "Generating right hand side\n";
  calc_b(b, A, x_soln, n); // Multiply matrix by solution to get the RHS, b
  std::fill(x, x + n, 0.0);

  // Solve
  auto start = high_resolution_clock::now();
  int iters;
  iters = cg_solve(x, A, b, n, cg_max_iter);
  auto stop = high_resolution_clock::now();
  auto duration =
      duration_cast<std::chrono::microseconds>(stop - start).count();

  std::cout << "Performed " << iters << " iterations" << std::endl;
  std::cout << "Solve time: " << duration << " us" << std::endl;
  std::cout << "Time per iteration: " << duration / iters << " us" << std::endl;

  real av_error2 = 0.0;
  for (int i = 0; i < n; ++i) {
    av_error2 += std::fabs(x_soln[i] - x[i]);
  }
  av_error2 /= n;

  std::cout << "Average error = " << std::sqrt(av_error2) << "\n";

  cudaFree(x);
  cudaFree(b);
  cudaFree(A);

  delete[] x_soln;

  return 0;
}
