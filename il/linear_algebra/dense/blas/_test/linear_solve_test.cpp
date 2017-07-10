//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/linear_algebra/dense/factorization/linearSolve.h>

TEST(linear_solve, square_matrix_0) {
  il::int_t test_passed{false};

  il::Array2D<double> A{3, 4, 0.0};
  for (il::int_t i = 0; i < 3; ++i) {
    A(i, i) = 1.0;
  }
  il::Array<double> y{3, 0.0};

  try {
    il::Status status{};
    il::Array<double> x{il::linearSolve(std::move(A), y, il::io, status)};
    status.ignoreError();
  } catch (il::AbortException) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, square_matrix_1) {
  il::int_t test_passed{false};

  il::Array2D<double> A{4, 3, 0.0};
  for (il::int_t i = 0; i < 3; ++i) {
    A(i, i) = 1.0;
  }
  il::Array<double> y{4, 0.0};

  try {
    il::Status status{};
    il::Array<double> x{il::linearSolve(std::move(A), y, il::io, status)};
    status.ignoreError();
  } catch (il::AbortException) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, size_y) {
  il::int_t test_passed{false};

  il::Array2D<double> A{3, 3, 0.0};
  for (il::int_t i = 0; i < 3; ++i) {
    A(i, i) = 1.0;
  }
  il::Array<double> y{4, 0.0};

  try {
    il::Status status{};
    il::Array<double> x{il::linearSolve(std::move(A), y, il::io, status)};
    status.ignoreError();
  } catch (il::AbortException) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, c_order) {
  il::Array2C<double> A{il::value, {{1.0, 2.0}, {1.0, 4.0}}};
  il::Array<double> y{il::value, {5.0, 9.0}};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  status.ignoreError();

  ASSERT_TRUE(x.size() == 2 && x[0] == 1.0 && x[1] == 2.0);
}

TEST(linear_solve, f_order) {
  il::Array2D<double> A{il::value, {{1.0, 1.0}, {2.0, 4.0}}};
  il::Array<double> y{il::value, {5.0, 9.0}};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  status.ignoreError();

  ASSERT_TRUE(x.size() == 2 && x[0] == 1.0 && x[1] == 2.0);
}

TEST(linear_solve, singular_matrix_0) {
  il::Array2D<double> A{2, 2, 0.0};
  il::Array<double> y{il::value, {1.0, 1.0}};
  bool test_passed{false};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  if (!status.ok() && status.error() == il::Error::MatrixSingular) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(linear_solve, singular_matrix_1) {
  il::Array2D<double> A{2, 2, 1.0};
  il::Array<double> y{il::value, {1.0, 1.0}};
  bool test_passed{false};

  il::Status status{};
  il::Array<double> x{il::linearSolve(A, y, il::io, status)};
  if (!status.ok() && status.error() == il::Error::MatrixSingular) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}
