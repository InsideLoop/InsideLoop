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

#include <il/Array2D.h>
#include <il/linear_algebra/dense/blas/blas.h>
#include <il/linear_algebra/matrixFree/solver/MatrixFreeGmres.h>

class DenseMatrix {
 private:
  il::Array2D<double> a_;

 public:
  DenseMatrix(il::Array2D<double> a) : a_{std::move(a)} {};
  il::int_t size(il::int_t d) const { return a_.size(d); };
  void dot(const il::ArrayView<double>& x, il::io_t,
           il::ArrayEdit<double>& y) const {
    IL_EXPECT_FAST(y.size() == a_.size(0));
    IL_EXPECT_FAST(x.size() == a_.size(1));

    for (il::int_t i = 0; i < y.size(); ++i) {
      y[i] = 0.0;
    }

    il::blas(1.0, a_.view(), x, 0.0, il::io, y);
  }
};

int main() {
  const il::int_t nb_eigen_values = 5;
  const il::int_t dim_eigen_spaces = 6;
  const il::int_t n = dim_eigen_spaces * nb_eigen_values;

  il::Array2D<double> a{n, n, 0.0};
  for (il::int_t k = 0; k < n; ++k) {
    a(k, k) = 1.0 + k / dim_eigen_spaces;
  }
  DenseMatrix m{a};
  const il::Array<double> y{n, 1.0};

  const double relative_precision_solver = 1.0e-5;
  const il::int_t max_nb_iterations = 100;
  const il::int_t restart_iteration = 20;
  // This object should be const
  il::MatrixFreeGmres<DenseMatrix> solver{relative_precision_solver,
                                          max_nb_iterations, restart_iteration};
  const il::Array<double> x = solver.solve(m, y);

  const il::int_t nb_iterations = solver.nbIterations();

  return 0;
}
