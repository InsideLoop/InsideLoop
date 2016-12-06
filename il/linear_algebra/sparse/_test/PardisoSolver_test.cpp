//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/linear_algebra/sparse/PardisoSolver.h>
#include <il/linear_algebra/sparse/_test/matrix/heat.h>

#include <il/benchmark/tools/timer/Timer.h>

TEST(PardisoSolver, base) {
  const double epsilon = 1.0e-15;

  il::Array<double> y{il::value, {3.0, 7.0}};

  il::Array<il::StaticArray<il::int_t, 2>> position{};
  position.emplace_back(il::value, std::initializer_list<il::int_t>{0, 0});
  position.emplace_back(il::value, std::initializer_list<il::int_t>{0, 1});
  position.emplace_back(il::value, std::initializer_list<il::int_t>{1, 0});
  position.emplace_back(il::value, std::initializer_list<il::int_t>{1, 1});

  il::Array<il::int_t> index{};
  il::SparseArray2C<double> A{2, position, il::io, index};
  A[index[0]] = 1.0;
  A[index[1]] = 2.0;
  A[index[2]] = 3.0;
  A[index[3]] = 4.0;

  il::PardisoSolver solver{};
  solver.symbolic_factorization(A);
  solver.numerical_factorization(A);

  il::Array<double> x = solver.solve(A, y);

  ASSERT_TRUE(il::abs(x[0] - 1.0) <= epsilon && il::abs(x[1] - 1.0) <= epsilon);
}

TEST(PardisoSolver, heat) {
  const il::int_t n = 10;

  il::SparseArray2C<double> A{};
  il::Array<double> y{};
  il::heat_1d(n, il::io, A, y);

  il::PardisoSolver solver{};
  solver.symbolic_factorization(A);
  solver.numerical_factorization(A);
  il::Array<double> x = solver.solve(A, y);

  double max_err = 0.0;
  for (il::int_t i = 0; i < n; ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  const double epsilon = 1.0e-15;
  ASSERT_TRUE(max_err <= epsilon);
}

TEST(PardisoSolver, heat2D) {
  const il::int_t n = 10;

  il::SparseArray2C<double> A{};
  il::Array<double> y{};
  il::heat_2d(n, il::io, A, y);

  il::PardisoSolver solver{};
  solver.symbolic_factorization(A);
  solver.numerical_factorization(A);
  il::Array<double> x = solver.solve(A, y);

  double max_err = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  const double epsilon = 1.0e-15;
  ASSERT_TRUE(max_err <= 10 * epsilon);
}

TEST(PardisoSolver, heat3D) {
  const il::int_t n = 10;

  il::SparseArray2C<double> A{};
  il::Array<double> y{};
  il::heat_3d(n, il::io, A, y);

  il::PardisoSolver solver{};
  solver.symbolic_factorization(A);
  solver.numerical_factorization(A);
  il::Array<double> x = solver.solve(A, y);


  double max_err = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  const double epsilon = 1.0e-7;
  ASSERT_TRUE(max_err <= 10 * epsilon);
}
