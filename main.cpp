//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/linear_algebra/dense/factorization/LU.h>

int main() {
  const il::int_t n = 2;
  il::Array2D<double> A{il::value, {{2.0, 0.0}, {0.0, 3.0}}};
  il::Array<double> y(n, 1.0);
// Fill y

  il::Status status{};
  il::LU<il::Array2D<double>> lu_decomposition(A, il::io, status);
  if (!status.ok()) {
    // The matrix is singular to the machine precision. You should deal with
    // the error.
  }

  il::Array<double> x = lu_decomposition.solve(y);

  return 0;
}
