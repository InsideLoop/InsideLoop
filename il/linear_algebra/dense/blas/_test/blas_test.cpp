//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/linear_algebra/dense/blas/blas.h>

TEST(Blas, daxpy) {
  il::Array<double> x{il::value, {1.0, 2.0}};
  il::Array<double> y{il::value, {1.0, 1.0}};

  // y <- 2.0 * x + y
  il::blas(2.0, x, il::io, y);

  ASSERT_TRUE(y.size() == 2 && y[0] == 3.0 && y[1] == 5.0);
}
