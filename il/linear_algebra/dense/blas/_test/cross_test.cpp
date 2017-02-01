//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/linear_algebra/dense/blas/cross.h>
#include <il/math.h>

TEST(cross, cross2_0) {
  il::StaticArray<double, 2> x{il::value, {2.0, 1.0}};

  il::StaticArray<double, 2> y = il::cross(x);

  ASSERT_TRUE(y[0] == -1.0 && y[1] == 2.0);
}

TEST(cross, cross3_0) {
  il::StaticArray<double, 3> x{il::value, {1.0, 2.0, 3.0}};
  il::StaticArray<double, 3> y{il::value, {4.0, 5.0, 6.0}};

  il::StaticArray<double, 3> z = il::cross(x, y);

  const double error = il::max(il::abs(z[0] - (-3.0)), il::abs(z[1] - 6.0),
                               il::abs(z[2] - (-3.0)));

  ASSERT_TRUE(error <= 1.0e-15);
}
