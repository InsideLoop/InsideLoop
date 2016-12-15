//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstdio>

#include <il/Timer.h>
#include <il/linear_algebra/factorization/Cholesky.h>
#include <il/linear_algebra/factorization/PartialLU.h>

int main() {
  const il::int_t n = 10000;
  il::Array2D<double> A{n, n};
  for (il::int_t j = 0; j < n; ++j) {
    for (il::int_t i = 0; i < n; ++i) {
      A(i, j) = 1.0 / (1 + il::abs(i - j));
    }
  }

  il::LowerArray2D<double> B{n};
  for (il::int_t j = 0; j < n; ++j) {
    for (il::int_t i = j; i < n; ++i) {
      B(i, j) = 1.0 / (1 + il::abs(i - j));
    }
  }

  il::Timer timer_llt{};
  il::Status status{};
  il::Cholesky<il::Array2D<double>> llt_decomposition{A, il::io, status};
  status.abort_on_error();
  timer_llt.stop();

  il::Timer timer_llt_lower{};
  il::Cholesky<il::LowerArray2D<double>> llt_lower_decomposition{B, il::io,
                                                                 status};
  status.abort_on_error();
  timer_llt_lower.stop();

  il::Timer timer_lu{};
  il::PartialLU<il::Array2D<double>> lu_decomposition{A, il::io, status};
  status.abort_on_error();
  timer_lu.stop();

  std::printf("Time for Cholesky decomposition: %7.2f s\n", timer_llt.time());
  std::printf("Time for Cholesky (packed) decomposition: %7.2f s\n",
              timer_llt_lower.time());
  std::printf("Time for LU decomposition: %7.2f s\n", timer_lu.time());

  return 0;
}
