//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/math.h>
#include <il/linear_algebra/dot.h>

TEST(dot, matrix_f_0) {
  il::Array2D<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}}};
  il::Array2D<double> B{il::value, {{5.0, 6.0}, {7.0, 8.0}}};

  il::Array2D<double> C{il::dot(A, B)};

  double error{
      il::max(il::abs(C(0, 0) - 23.0) / 23.0, il::abs(C(1, 0) - 34.0) / 34.0,
              il::abs(C(0, 1) - 31.0) / 31.0, il::abs(C(1, 1) - 46.0) / 46.0)};

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_f_1) {
  il::Array2D<double> A{il::value, {{1.0}, {2.0}}};
  il::Array2D<double> B{il::value, {{3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}}};

  il::Array2D<double> C{il::dot(A, B)};

  double error{il::max(il::abs(C(0, 0) - 11.0) / 11.0,
                       il::abs(C(0, 1) - 17.0) / 17.0,
                       il::abs(C(0, 2) - 23.0) / 23.0)};

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_f_simd_0) {
  il::Array2D<double> A{2, 2, il::align, il::simd};
  A(1, 0) = 2.0;
  A(0, 1) = 3.0;
  A(1, 1) = 4.0;
  il::Array2D<double> B{2, 2, il::align, il::simd};
  B(0, 0) = 5.0;
  B(1, 0) = 6.0;
  B(0, 1) = 7.0;
  B(1, 1) = 8.0;

  il::Array2D<double> C{il::dot(A, B)};

  double error{
      il::max(il::abs(C(0, 0) - 23.0) / 23.0, il::abs(C(1, 0) - 34.0) / 34.0,
              il::abs(C(0, 1) - 31.0) / 31.0, il::abs(C(1, 1) - 46.0) / 46.0)};

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_f_simd_1) {
  il::Array2D<double> A{1, 2, il::align, il::simd};
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  il::Array2D<double> B{2, 3, il::align, il::simd};
  B(0, 0) = 3.0;
  B(1, 0) = 4.0;
  B(0, 1) = 5.0;
  B(1, 1) = 6.0;
  B(0, 2) = 7.0;
  B(1, 2) = 8.0;

  il::Array2D<double> C{il::dot(A, B)};

  double error{il::max(il::abs(C(0, 0) - 11.0) / 11.0,
                       il::abs(C(0, 1) - 17.0) / 17.0,
                       il::abs(C(0, 2) - 23.0) / 23.0)};

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_c_0) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}}};
  il::Array2C<double> B{il::value, {{5.0, 6.0}, {7.0, 8.0}}};

  il::Array2C<double> C{il::dot(A, B)};

  double error{
      il::max(il::abs(C(0, 0) - 19.0) / 19.0, il::abs(C(0, 1) - 22.0) / 22.0,
              il::abs(C(1, 0) - 43.0) / 43.0, il::abs(C(1, 1) - 50.0) / 50.0)};

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_c_1) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}}};
  il::Array2C<double> B{il::value, {{3.0, 5.0, 7.0}, {4.0, 6.0, 8.0}}};

  il::Array2C<double> C{il::dot(A, B)};

  double error{il::max(il::abs(C(0, 0) - 11.0) / 11.0,
                       il::abs(C(0, 1) - 17.0) / 17.0,
                       il::abs(C(0, 2) - 23.0) / 23.0)};

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}

TEST(dot, matrix_c_simd_0) {
  il::Array2C<double> A{2, 2, il::align, il::simd};
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  A(1, 0) = 3.0;
  A(1, 1) = 4.0;
  il::Array2C<double> B{2, 2, il::align, il::simd};
  B(0, 0) = 5.0;
  B(0, 1) = 6.0;
  B(1, 0) = 7.0;
  B(1, 1) = 8.0;

  il::Array2C<double> C{il::dot(A, B)};

  double error{
      il::max(il::abs(C(0, 0) - 19.0) / 19.0, il::abs(C(0, 1) - 22.0) / 22.0,
              il::abs(C(1, 0) - 43.0) / 43.0, il::abs(C(1, 1) - 50.0) / 50.0)};

  ASSERT_TRUE(C.size(0) == 2 && C.size(1) == 2 && error <= 1.0e-15);
}

TEST(dot, matrix_c_simd_1) {
  il::Array2C<double> A{1, 2, il::align, il::simd};
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  il::Array2C<double> B{2, 3, il::align, il::simd};
  B(0, 0) = 3.0;
  B(0, 1) = 5.0;
  B(0, 2) = 7.0;
  B(1, 0) = 4.0;
  B(1, 1) = 6.0;
  B(1, 2) = 8.0;

  il::Array2C<double> C{il::dot(A, B)};

  double error{il::max(il::abs(C(0, 0) - 11.0) / 11.0,
                       il::abs(C(0, 1) - 17.0) / 17.0,
                       il::abs(C(0, 2) - 23.0) / 23.0)};

  ASSERT_TRUE(C.size(0) == 1 && C.size(1) == 3 && error <= 1.0e-15);
}
