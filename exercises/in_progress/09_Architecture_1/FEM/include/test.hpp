/**
 * @file test.hpp
 * @brief Analytic validation for the dense FEM heat solver.
 */

#pragma once

#include "utils.hpp"

void validate_solution(const float *temperature, const Parameters &params);
