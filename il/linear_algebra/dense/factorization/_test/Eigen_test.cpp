//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <algorithm>

#include <gtest/gtest.h>

#include <il/Array2D.h>
#include <il/linear_algebra/dense/factorization/Eigen.h>

bool complex_sort(std::complex<double> z0, std::complex<double> z1) {
  if (std::real(z0) == std::real(z1)) {
    return std::imag(z0) < std::imag(z1);
  } else {
    return std::real(z0) < std::real(z1);
  }
}

TEST(Eigen, test0) {
  il::Array2D<double> A{il::value,
                        {{0.0, 3.0, -2.0}, {2.0, -2.0, 2.0}, {-1.0, 0.0, 1.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.abortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.begin(), ev.end(), complex_sort);
  il::Array<std::complex<double>> result{il::value,
                                         {{-4, 0.0}, {1.0, 0.0}, {2.0, 0.0}}};
  const double epsilon = 1.0e-15;

  ASSERT_TRUE(ev.size() == 3 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon);
}

TEST(Eigen, test1) {
  il::Array2D<double> A{il::value,
                        {{1.0, 0.0, -1.0}, {4.0, 6.0, 4.0}, {-2.0, -3.0, 0.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.abortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.begin(), ev.end(), complex_sort);
  il::Array<std::complex<double>> result{il::value,
                                         {{2.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}}};
  const double epsilon = 1.0e-5;

  ASSERT_TRUE(ev.size() == 3 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon);
}

TEST(Eigen, test2) {
  il::Array2D<double> A{il::value,
                        {{2.0, 5.0, -3.0}, {2.0, 1.0, 4.0}, {-3.0, -5.0, 0.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.abortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.begin(), ev.end(), complex_sort);
  il::Array<std::complex<double>> result{il::value,
                                         {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}}};
  const double epsilon = 1.0e-4;
  ASSERT_TRUE(ev.size() == 3 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon);
}

TEST(Eigen, test3) {
  il::Array2D<double> A{il::value,
                        {{0.0, 0.0, 0.0, -1.0},
                         {0.0, 0.0, 1.0, 0.0},
                         {0.0, -1.0, 0.0, 0.0},
                         {1, 0.0, 0.0, 0.0}}};

  il::Status status{};
  il::Eigen<il::Array2D<double>> eigen_decomposition{A, il::io, status};
  status.abortOnError();

  il::Array<std::complex<double>> ev = eigen_decomposition.eigenValue();
  std::sort(ev.begin(), ev.end(), complex_sort);
  il::Array<std::complex<double>> result{
      il::value, {{0.0, -1.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 1.0}}};
  const double epsilon = 1.0e-15;

  ASSERT_TRUE(ev.size() == 4 && std::abs(ev[0] - result[0]) <= epsilon &&
              std::abs(ev[1] - result[1]) <= epsilon &&
              std::abs(ev[2] - result[2]) <= epsilon &&
              std::abs(ev[3] - result[3]) <= epsilon);
}
