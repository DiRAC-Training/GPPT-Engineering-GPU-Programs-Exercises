#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "idx.hpp"
#include "precision.hpp"
#include "solver.hpp"

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

const bool RANDOMISE_SEED = true;

// Diagonal shift that makes the matrix symmetric positive definite. DIAG_SCALE
// > 0 uses f*sqrt(n), giving a poorly conditioned (but reliably solvable)
// system that takes more CG iterations to converge; otherwise n/32, which is
// better conditioned and converges in fewer iterations.
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
  const int cg_max_iter = 128; // headroom: this system converges in ~67 iters

  // Host buffers. A_h and b_h are initialised on the host and never come back
  // from the device; x_h holds the initial guess going in and receives the
  // solution from the device coming back.
  real *A_h = new real[n * n]; // note n*n
  real *b_h = new real[n];
  real *x_h = new real[n];

  // The reference solution is only used to compare the final result. It never
  // goes to the device, so it carries no suffix.
  real *x_soln = new real[n];

  init_rng(RANDOMISE_SEED);

  // Initial conditions
  generate_positive_definite(A_h, n); // Generate positive def matrix
  std::cout << "Generating random solution\n";
  fill_rand_vec(x_soln, n, -1.0, 1.0); // Generate a random solution vector
  std::cout << "Generating right hand side\n";
  calc_b(b_h, A_h, x_soln, n); // Multiply matrix by solution to get the RHS, b
  std::fill(x_h, x_h + n, 0.0);

#ifdef USE_GPU
  // Device copies of the three arrays the kernels operate on. The host has
  // initialised A and b, so copy them across once here.
  real *A_d, *b_d, *x_d;
  cudaMalloc(&A_d, (size_t)n * n * sizeof(real));
  cudaMalloc(&b_d, n * sizeof(real));
  cudaMalloc(&x_d, n * sizeof(real));
  cudaMemcpy(A_d, A_h, (size_t)n * n * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, n * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, n * sizeof(real), cudaMemcpyHostToDevice);
#endif

  // Solve
  auto start = high_resolution_clock::now();
  int iters;
#ifdef USE_GPU
  iters = cg_solve(x_d, A_d, b_d, n, cg_max_iter);
#else
  iters = cg_solve(x_h, A_h, b_h, n, cg_max_iter);
#endif
  auto stop = high_resolution_clock::now();
  auto duration =
      duration_cast<std::chrono::microseconds>(stop - start).count();

#ifdef USE_GPU
  // Bring the solution back to the host for the error check, then release the
  // device buffers.
  cudaMemcpy(x_h, x_d, n * sizeof(real), cudaMemcpyDeviceToHost);
  cudaFree(x_d);
  cudaFree(b_d);
  cudaFree(A_d);
#endif

  std::cout << "Performed " << iters << " iterations" << std::endl;
  std::cout << "Solve time: " << duration << " us" << std::endl;
  std::cout << "Time per iteration: " << duration / iters << " us" << std::endl;

  real av_error2 = 0.0;
  for (int i = 0; i < n; ++i) {
    av_error2 += std::fabs(x_soln[i] - x_h[i]);
  }
  av_error2 /= n;

  std::cout << "Average error = " << std::sqrt(av_error2) << "\n";

  delete[] A_h;
  delete[] b_h;
  delete[] x_h;
  delete[] x_soln;

  return 0;
}
