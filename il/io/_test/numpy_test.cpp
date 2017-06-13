//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/io/numpy.h>

il::String filename = "/Users/fayard/Desktop/b.npy";

TEST(numpy, array_0) {
  il::Array<int> v{il::value, {1, 2, 3}};

  il::Status save_status{};
  il::save(v, filename, il::io, save_status);

  il::Status load_status{};
  il::Array<int> w = il::load<il::Array<int>>(filename, il::io, load_status);

  ASSERT_TRUE(!save_status.not_ok() && !load_status.not_ok() && w.size() == 3 &&
              w[0] == v[0] && w[1] == v[1] && w[2] == v[2]);
}

TEST(numpy, array_1) {
  il::Array<double> v{il::value, {1.0, 2.0, 3.0}};

  il::Status save_status{};
  il::save(v, filename, il::io, save_status);

  il::Status load_status{};
  il::Array<double> w =
      il::load<il::Array<double>>(filename, il::io, load_status);

  ASSERT_TRUE(!save_status.not_ok() && !load_status.not_ok() && w.size() == 3 &&
              w[0] == v[0] && w[1] == v[1] && w[2] == v[2]);
}

TEST(numpy, array2d_0) {
  il::Array2D<int> A{il::value, {{1, 2}, {3, 4}, {5, 6}}};

  il::Status save_status{};
  il::save(A, filename, il::io, save_status);

  il::Status load_status{};
  il::Array2D<int> B =
      il::load<il::Array2D<int>>(filename, il::io, load_status);

  ASSERT_TRUE(!save_status.not_ok() && !load_status.not_ok() &&
              B.size(0) == A.size(0) && B.size(1) == A.size(1) &&
              B(0, 0) == A(0, 0) && B(1, 0) == A(1, 0) && B(0, 1) == A(0, 1) &&
              B(1, 1) == A(1, 1) && B(0, 2) == A(0, 2) && B(1, 2) == A(1, 2));
}

TEST(numpy, array2d_1) {
  il::Array2D<double> A{il::value, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};

  il::Status save_status{};
  il::save(A, filename, il::io, save_status);

  il::Status load_status{};
  il::Array2D<double> B =
      il::load<il::Array2D<double>>(filename, il::io, load_status);

  ASSERT_TRUE(!save_status.not_ok() && !load_status.not_ok() &&
              B.size(0) == A.size(0) && B.size(1) == A.size(1) &&
              B(0, 0) == A(0, 0) && B(1, 0) == A(1, 0) && B(0, 1) == A(0, 1) &&
              B(1, 1) == A(1, 1) && B(0, 2) == A(0, 2) && B(1, 2) == A(1, 2));
}
