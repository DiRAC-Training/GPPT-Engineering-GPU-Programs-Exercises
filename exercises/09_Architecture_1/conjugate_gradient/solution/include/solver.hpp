#pragma once

#include "precision.hpp"

// cg_solve is shared between both backends: it drives the CG iteration entirely
// through the primitives in kernels.hpp and never dereferences the arrays it is
// given, so the pointers may live in host or device memory depending on how the
// project was built.
int cg_solve(real *x, const real *A, const real *b, const int n, int max_iter);
