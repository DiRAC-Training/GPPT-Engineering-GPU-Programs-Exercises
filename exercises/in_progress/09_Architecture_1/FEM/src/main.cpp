/**
 * @file main.cpp
 * @brief Driver for the dense FEM heat solver.
 *
 * Populates Parameters, assembles the system, solves with CG, validates
 * the numerical solution, and writes output.
 */

#include <chrono>
#include <iostream>

#include "solver.hpp"
#include "test.hpp"
#include "utils.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    // Populate parameters
    Parameters params;
    params.elements_per_side = 50;
    params.domain_length     = 1.0;
    params.source_strength   = 100.0;
    params.source_x_min      = 0.4;
    params.source_x_max      = 0.6;
    params.source_y_min      = 0.4;
    params.source_y_max      = 0.6;
    params.t_boundary        = 0.0;

    const int cg_max_iter = 10000;
    const float cg_tol     = 1e-6;

    int n_nodes        = params.n_nodes();
    int nodes_per_side = params.nodes_per_side();

    std::cout << "2D FEM Heat Equation" << std::endl;
    std::cout << "Grid: " << nodes_per_side << " x " << nodes_per_side
              << " nodes (" << params.elements_per_side << " x "
              << params.elements_per_side << " elements)" << std::endl;

    // Allocate memory for stiffness matrix, RHS and solution
    DenseMatrix A;
    A.allocate(n_nodes);
    float *rhs = new float[n_nodes];
    float *temperature = new float[n_nodes]();
     
    // Assemble stiffness matrix and RHS
    assemble(A, rhs, params);

    // Apply boundary conditions: T = t_boundary on all edges
    apply_boundary_conditions(A, rhs, params);

    // Solve
    auto start    = high_resolution_clock::now();
    int iters     = cg_solve(A, rhs, temperature, cg_max_iter, cg_tol);
    auto stop     = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start).count();

    std::cout << "CG converged in " << iters << " iterations" << std::endl;
    std::cout << "Solve time: " << duration << " ms" << std::endl;

    // Analytic validation
    validate_solution(temperature, params);

    // Output final result
    write_solution(temperature, "temperature.dat", params);

    // Clean up
    A.free();
    delete[] rhs;
    delete[] temperature;

    return 0;
}
