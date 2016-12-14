//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/linear_algebra/factorization/PartialLU.h>

int main() {
  il::Array2D<double> A{il::value, {{2.0, 2.0}, {2.0, 2.0}}};

  il::Error error{};
  il::PartialLU<il::Array2D<double>> lu_decomposition{std::move(A), il::io,
                                                      error};
  error.abort();

  il::Array2D<double> A_inv = lu_decomposition.inverse();

  return 0;
}
