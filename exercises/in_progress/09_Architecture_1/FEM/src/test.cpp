/**
 * @file test.cpp
 * @brief Analytic validation for the dense FEM heat solver.
 *
 * Compares the numerical solution against the exact double Fourier
 * series solution for -nabla^2 T = f on a unit square with T = 0 on
 * all boundaries and a localised source patch.
 */

#include "test.hpp"

#include <cmath>
#include <iostream>

// ---------------------------------------------------------------------------
// Analytic solution
// ---------------------------------------------------------------------------

const int num_terms = 50;

/**
 * @brief Evaluate the analytic solution at a point (x, y).
 *
 * Double Fourier sine series for -nabla^2 T = f on [0,1]^2 with T = 0
 * on all boundaries and constant source f over a rectangular patch.
 */
static double analytic_solution(double x, double y, double f,
                                double x1, double x2, double y1, double y2) {
    double sum = 0.0;
    double pi  = M_PI;
    double pi4 = pi * pi * pi * pi;

    for (int m = 1; m <= num_terms; m++) {
        for (int n = 1; n <= num_terms; n++) {
            double denom = pi4 * m * n * (m * m + n * n);
            double coeff = (std::cos(m * pi * x1) - std::cos(m * pi * x2)) *
                           (std::cos(n * pi * y1) - std::cos(n * pi * y2));
            sum += coeff * std::sin(m * pi * x) * std::sin(n * pi * y) / denom;
        }
    }

    return 4.0 * f * sum;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * @brief Validate the numerical solution against the analytic Fourier series.
 *
 * Computes L2 and L-infinity errors over interior nodes and compares the
 * centre-point value against the analytic result. Prints all results.
 */
void validate_solution(const float *temperature, const Parameters &params) {
    int nodes_per_side = params.nodes_per_side();
    float h = params.element_spacing();

    // L2 and L-infinity errors over interior nodes
    float sum_sq        = 0.0;
    float linf_error    = 0.0;
    int interior_count = 0;

    for (int ix = 1; ix < nodes_per_side - 1; ix++) {
        for (int iy = 1; iy < nodes_per_side - 1; iy++) {
            double x     = ix * h;
            double y     = iy * h;
            double exact = analytic_solution(x, y, params.source_strength,
                                             params.source_x_min,
                                             params.source_x_max,
                                             params.source_y_min,
                                             params.source_y_max);
            float err = std::abs(temperature[node_idx(ix, iy, nodes_per_side)] -
                                static_cast<float>(exact));
            sum_sq += err * err;
            linf_error = std::max(linf_error, err);
            interior_count++;
        }
    }

    float l2_error = std::sqrt(sum_sq / interior_count);

    std::cout << "L2 error (interior):    " << l2_error << std::endl;
    std::cout << "L-inf error (interior): " << linf_error << std::endl;

    // Centre-point comparison
    float centre_numerical =
        temperature[node_idx(nodes_per_side / 2, nodes_per_side / 2,
                             nodes_per_side)];
    double centre_analytic = analytic_solution(
        params.domain_length / 2.0, params.domain_length / 2.0,
        params.source_strength, params.source_x_min, params.source_x_max,
        params.source_y_min, params.source_y_max);
    std::cout << "Centre numerical:  " << centre_numerical << std::endl;
    std::cout << "Centre analytic:   " << centre_analytic << std::endl;
    std::cout << "Centre error:      "
              << std::abs(centre_numerical - static_cast<float>(centre_analytic))
              << std::endl;
}
