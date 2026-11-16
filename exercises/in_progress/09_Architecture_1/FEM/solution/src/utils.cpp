/**
 * @file utils.cpp
 * @brief Implementation of setup functions for the dense FEM heat solver.
 */

#include "utils.hpp"

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/**
 * @brief Assemble the global stiffness matrix and right-hand side vector.
 *
 * Loops over all elements and scatters each element stiffness contribution
 * into the dense global matrix. Simultaneously assembles the consistent
 * load vector for a localised source patch.
 */
void assemble(DenseMatrix &A, float *rhs, const Parameters &params) {
    int n_nodes = params.n_nodes();
    int elements_per_side = params.elements_per_side;
    int nodes_per_side = params.nodes_per_side();
    float h = params.element_spacing();

    std::memset(A.vals, 0, A.size * A.size * sizeof(float));
    std::memset(rhs, 0, n_nodes * sizeof(float));
    float load_per_node = params.source_strength * h * h / 4.0;

#pragma omp parallel for collapse(2)
    for (int ex = 0; ex < elements_per_side; ex++) {
        for (int ey = 0; ey < elements_per_side; ey++) {
            for (int a = 0; a < 4; a++) {
                for (int b = 0; b < 4; b++) {
#pragma omp atomic
                    A(local_to_global(ex, ey, a, nodes_per_side),
                      local_to_global(ex, ey, b, nodes_per_side)) +=
                        element_stiffness[a][b];
                }
            }

            float cx = (ex + 0.5) * h;
            float cy = (ey + 0.5) * h;
            if (cx >= params.source_x_min && cx <= params.source_x_max &&
                cy >= params.source_y_min && cy <= params.source_y_max) {
                for (int a = 0; a < 4; a++) {
#pragma omp atomic
                    rhs[local_to_global(ex, ey, a, nodes_per_side)] +=
                        load_per_node;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

/**
 * @brief Apply boundary conditions (T = t_boundary on all edges).
 *
 * Identifies boundary nodes on the grid edges, then uses symmetric
 * elimination to modify the matrix and RHS: known boundary values are
 * moved to the right-hand side, boundary rows/columns are zeroed, and
 * the diagonal is set to 1.
 */
void apply_boundary_conditions(DenseMatrix &A, float *rhs,
                               const Parameters &params) {
    int n_nodes = params.n_nodes();
    int nodes_per_side = params.nodes_per_side();

    bool *is_boundary = new bool[n_nodes]();
    float *bc_vals     = new float[n_nodes]();

    for (int ix = 0; ix < nodes_per_side; ix++) {
        is_boundary[node_idx(ix, 0, nodes_per_side)]                  = true;
        is_boundary[node_idx(ix, nodes_per_side - 1, nodes_per_side)] = true;
        bc_vals[node_idx(ix, 0, nodes_per_side)]                  = params.t_boundary;
        bc_vals[node_idx(ix, nodes_per_side - 1, nodes_per_side)] = params.t_boundary;
    }
    for (int iy = 1; iy < nodes_per_side - 1; iy++) {
        is_boundary[node_idx(0, iy, nodes_per_side)]                  = true;
        is_boundary[node_idx(nodes_per_side - 1, iy, nodes_per_side)] = true;
        bc_vals[node_idx(0, iy, nodes_per_side)]                  = params.t_boundary;
        bc_vals[node_idx(nodes_per_side - 1, iy, nodes_per_side)] = params.t_boundary;
    }

    // Symmetric elimination
#pragma omp parallel for
    for (int i = 0; i < A.size; i++) {
        if (is_boundary[i])
            continue;
        for (int j = 0; j < A.size; j++) {
            if (is_boundary[j]) {
                rhs[i] -= A(i, j) * bc_vals[j];
                A(i, j) = 0.0;
            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < A.size; i++) {
        if (!is_boundary[i])
            continue;
        for (int j = 0; j < A.size; j++)
            A(i, j) = 0.0;
        A(i, i) = 1.0;
        rhs[i]  = bc_vals[i];
    }

    delete[] is_boundary;
    delete[] bc_vals;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/**
 * @brief Write the solution to a plain text file (x y T per line).
 */
void write_solution(const float *temperature, const char *filename,
                    const Parameters &params) {
    int nodes_per_side = params.nodes_per_side();
    float h = params.element_spacing();

    std::ofstream out(filename);
    for (int ix = 0; ix < nodes_per_side; ix++)
        for (int iy = 0; iy < nodes_per_side; iy++)
            out << ix * h << " " << iy * h << " "
                << temperature[node_idx(ix, iy, nodes_per_side)] << "\n";
    out.close();
    std::cout << "Solution written to " << filename << std::endl;
}
