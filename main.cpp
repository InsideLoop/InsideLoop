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

#include <iostream>

#include <il/Array.h>
#include <il/linear_algebra/matrixFree/solver/Gmres.h>

class Diagonal : public il::ArrayFunctor<double> {
 private:
  il::int_t n_;
  double epsilon_;

 public:
  Diagonal(il::int_t n, double epsilon) {
    n_ = n;
    epsilon_ = epsilon;
  }
  il::int_t sizeInput() const override { return n_; }
  il::int_t sizeOutput() const override { return n_; }
  void operator()(il::ArrayView<double> x, il::io_t,
                  il::ArrayEdit<double> y) const override {
    IL_EXPECT_FAST(x.size() == n_);
    IL_EXPECT_FAST(y.size() == n_);

    const il::int_t n = n_;
    const double epsilon = epsilon_;
    const double alpha = 1.0 / (n - 1);
    y[0] = x[0];
    for (il::int_t i = 1; i < n - 1; ++i) {
      y[i] = epsilon * x[i - 1] + (1 + alpha * i) * x[i] + epsilon * x[i + 1];
    }
    y[n - 1] = 2 * x[n - 1];
  }
};

class Preconditionner : public il::ArrayFunctor<double> {
 private:
  il::int_t n_;

 public:
  Preconditionner(il::int_t n) { n_ = n; }
  il::int_t sizeInput() const override { return n_; }
  il::int_t sizeOutput() const override { return n_; }
  void operator()(il::ArrayView<double> x, il::io_t,
                  il::ArrayEdit<double> y) const override {
    IL_EXPECT_FAST(x.size() == n_);
    IL_EXPECT_FAST(y.size() == n_);

    const il::int_t n = n_;
    const double alpha = 1.0 / (n - 1);
    for (il::int_t i = 0; i < n; ++i) {
      y[i] = x[i] / (1 + alpha * i);
    }
  }
};

int main() {
  const il::int_t n = 100;
  const double epsilon = 0.1;
  const Diagonal matrix{n, epsilon};
  const Preconditionner preconditionner{n};

  const double relative_precision = 1.0e-1;
  const il::int_t max_nb_iterations = 100;
  const il::int_t restart_iteration = 10;

  il::Gmres gmres_solver{relative_precision, max_nb_iterations,
                         restart_iteration};
  const il::Array<double> y{n, 1.0};
  il::Array<double> x{n};

  const bool use_preconditionner = true;
  const bool use_x_as_initial_value = false;
  gmres_solver.solve(matrix, preconditionner, y.view(), use_preconditionner,
                     use_x_as_initial_value, il::io, x.edit());

  std::cout << "Number of iterations: " << gmres_solver.nbIterations()
            << std::endl;

  return 0;
}
