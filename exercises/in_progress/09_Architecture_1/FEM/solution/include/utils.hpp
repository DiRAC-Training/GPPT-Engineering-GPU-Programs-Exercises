/**
 * @file utils.hpp
 * @brief Data structures and setup functions for the dense FEM heat solver.
 *
 * Contains the Parameters struct, DenseMatrix, assembly, boundary condition
 * application, and output routines. The DenseMatrix allocation switches
 * between host (`new`) and CUDA managed memory (`cudaMallocManaged`) based
 * on the USE_GPU preprocessor symbol set by the build system.
 */

#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Problem and solver parameters, populated in main.
struct Parameters {
    int elements_per_side;
    float domain_length;
    float source_strength;
    float source_x_min, source_x_max;
    float source_y_min, source_y_max;
    float t_boundary;

    int nodes_per_side() const { return elements_per_side + 1; }
    int n_nodes() const { return nodes_per_side() * nodes_per_side(); }
    float element_spacing() const { return domain_length / elements_per_side; }
};

// ---------------------------------------------------------------------------
// Dense matrix
// ---------------------------------------------------------------------------

/// Dense square matrix stored in row-major order. Storage is allocated on
/// the host for CPU builds and as CUDA managed memory for GPU builds.
struct DenseMatrix {
    int size;   ///< Matrix dimension (size x size)
    float *vals; ///< Row-major storage, length size*size

    // Prevent accidental copies of the matrix
    DenseMatrix(const DenseMatrix&) = delete;
    DenseMatrix& operator=(const DenseMatrix&) = delete;

    void allocate(int n) {
        size = n;
#ifdef USE_GPU
        cudaMallocManaged(&vals, (size_t)size * size * sizeof(float));
        std::memset(vals, 0, (size_t)size * size * sizeof(float));
#else
        vals = new float[size * size]();
#endif
    }

    void free() {
#ifdef USE_GPU
        cudaFree(vals);
#else
        delete[] vals;
#endif
        vals = nullptr;
        size = 0;
    }

    float &operator()(int i, int j) { return vals[i * size + j]; }
    float operator()(int i, int j) const { return vals[i * size + j]; }
};

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

/// Bilinear quad element stiffness matrix for the Laplacian on a square grid.
/// Node ordering: 0=SW, 1=SE, 2=NE, 3=NW (counter-clockwise).
const float element_stiffness[4][4] = {
    {2.0 / 3.0, -1.0 / 6.0, -1.0 / 3.0, -1.0 / 6.0},
    {-1.0 / 6.0, 2.0 / 3.0, -1.0 / 6.0, -1.0 / 3.0},
    {-1.0 / 3.0, -1.0 / 6.0, 2.0 / 3.0, -1.0 / 6.0},
    {-1.0 / 6.0, -1.0 / 3.0, -1.0 / 6.0, 2.0 / 3.0}};

/// Grid offsets for each local node: 0=SW, 1=SE, 2=NE, 3=NW.
const int local_offset_x[4] = {0, 1, 1, 0};
const int local_offset_y[4] = {0, 0, 1, 1};

/// Compute the global node index from grid coordinates (row-major order).
inline int node_idx(const int ix, const int iy, const int nodes_per_side) {
    return ix * nodes_per_side + iy;
}

/// Map local node index a on element (ex, ey) to a global node index.
inline int local_to_global(const int ex, const int ey, const int a,
                           const int nodes_per_side) {
    return node_idx(ex + local_offset_x[a], ey + local_offset_y[a],
                    nodes_per_side);
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

void assemble(DenseMatrix &A, float *rhs, const Parameters &params);

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

void apply_boundary_conditions(DenseMatrix &A, float *rhs,
                               const Parameters &params);

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void write_solution(const float *temperature, const char *filename,
                    const Parameters &params);
