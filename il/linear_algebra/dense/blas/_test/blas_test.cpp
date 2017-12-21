//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/linear_algebra/dense/blas/blas.h>

#ifdef IL_BLAS
TEST(Blas, daxpy) {
  il::Array<double> x{il::value, {1.0, 2.0}};
  il::Array<double> y{il::value, {1.0, 1.0}};

  // y <- 2.0 * x + y
  il::blas(2.0, x, il::io, y);

  ASSERT_TRUE(y.size() == 2 && y[0] == 3.0 && y[1] == 5.0);
}

TEST(Blas, Array2DDoubleArray2DDouble_0) {
  il::Array2D<double> A{il::value, {{1.0}, {2.0}}};
  il::Array2D<double> B{il::value, {{3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}}};
  const double alpha = 2.0;
  const double beta = 3.0;
  il::Array2D<double> C{il::value, {{9.0}, {10.0}, {11.0}}};

  il::blas(alpha, A, B, beta, il::io, C);

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && C(0, 0) == 49.0 &&
              C(0, 1) == 64.0 && C(0, 2) == 79.0);
}

TEST(Blas, Array2DDoubleArray2DDouble_1) {
  il::Array2D<double> A{il::value, {{1.0}, {2.0}}};
  il::Array2D<double> B{il::value, {{3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}}};
  const double alpha = 2.0;
  const double beta = 3.0;
  il::Array2D<double> C{il::value, {{9.0}, {10.0}, {11.0}}};
  A.reserve(A.size(0) + 3, A.size(1) + 4);
  B.reserve(B.size(0) + 5, B.size(1) + 6);
  C.reserve(C.size(0) + 7, C.size(1) + 8);

  il::blas(alpha, A, B, beta, il::io, C);

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && C(0, 0) == 49.0 &&
      C(0, 1) == 64.0 && C(0, 2) == 79.0);
}
#endif
