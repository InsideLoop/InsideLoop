//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/linear_algebra/sparse/factorization/GmresIlu0.h>
#include <il/linear_algebra/sparse/factorization/_test/matrix/heat.h>

#include <il/benchmark/tools/timer/Timer.h>

TEST(GmresIlu0, base) {
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

  const double relative_precision = epsilon;
  const il::int_t max_nb_iteration = 10;
  const il::int_t nb_restart = 5;
  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
  solver.compute_preconditionner(il::io, A);
  il::Array<double> x = solver.solve(y, il::io, A);

  ASSERT_TRUE(il::abs(x[0] - 1.0) <= epsilon && il::abs(x[1] - 1.0) <= epsilon);
}

TEST(GmresIlu0, heat) {
  const il::int_t n = 10;

  il::SparseArray2C<double> A{};
  il::Array<double> y{};
  il::heat_1d(n, il::io, A, y);

  const double relative_precision = 1.0e-15;
  const il::int_t max_nb_iteration = 10;
  const il::int_t nb_restart = 5;
  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
  solver.compute_preconditionner(il::io, A);
  il::Array<double> x = solver.solve(y, il::io, A);

  double max_err = 0.0;
  for (il::int_t i = 0; i < n; ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  ASSERT_TRUE(max_err <= 10 * relative_precision);
}

TEST(GmresIlu0, heat2D) {
  const il::int_t n = 10;

  il::SparseArray2C<double> A{};
  il::Array<double> y{};
  il::heat_2d(n, il::io, A, y);

  const double relative_precision = 1.0e-7;
  const il::int_t max_nb_iteration = 100;
  const il::int_t nb_restart = 10;
  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
  solver.compute_preconditionner(il::io, A);
  il::Array<double> x = solver.solve(y, il::io, A);
  il::int_t nb_iteration = solver.nb_iteration();
  (void)nb_iteration;

  double max_err = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  ASSERT_TRUE(max_err <= 10 * relative_precision);
}

TEST(GmresIlu0, heat3D) {
  const il::int_t n = 10;

  il::SparseArray2C<double> A{};
  il::Array<double> y{};
  il::heat_3d(n, il::io, A, y);

  const double relative_precision = 1.0e-7;
  const il::int_t max_nb_iteration = 100;
  const il::int_t nb_restart = 10;
  il::GmresIlu0 solver{relative_precision, max_nb_iteration, nb_restart};
  solver.compute_preconditionner(il::io, A);
  il::Array<double> x = solver.solve(y, il::io, A);

  double max_err = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    max_err = il::max(max_err, il::abs(x[i] - 1.0));
  }

  ASSERT_TRUE(max_err <= 10 * relative_precision);
}
