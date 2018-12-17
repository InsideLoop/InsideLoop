//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
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

#include <il/Array.h>
#include <il/Dynamic.h>
#include <il/Gmres.h>
#include <il/Map.h>

template <typename T>
class Matrix : public il::FunctorArray<T> {
 private:
  il::Array<T> diag_;

 public:
  Matrix(il::Array<T> diag) : diag_{std::move(diag)} {}
  il::int_t size(il::int_t d) const override {
    IL_EXPECT_MEDIUM(d == 0 || d == 1);
    return diag_.size();
  }
  void operator()(il::ArrayView<T> x, il::io_t,
                  il::ArrayEdit<T> y) const override {
    const il::int_t n = diag_.size();
    IL_EXPECT_FAST(x.size() == n);
    IL_EXPECT_FAST(y.size() == n);

    for (il::int_t i = 0; i < n; ++i) {
      y[i] = diag_[i] * x[i];
    }
  }
};

int main() {
  const il::int_t n = 10;
  const double epsilon = 1.0e-2;
  il::Array<double> diag_m{n};
  il::Array<double> diag_cond{n};
  il::Array<double> y{n};
  for (il::int_t i = 0; i < n; ++i) {
    diag_m[i] = 2.0 + i + epsilon;
    diag_cond[i] = 1.0 / (2.0 + i);
    y[i] = 2.0 + i;
  }
  Matrix<double> m{diag_m};
  Matrix<double> cond{diag_cond};

  const il::int_t restart_iteration = 5;
  il::Gmres<double> gmres{m, cond, restart_iteration};

  const double relative_precision = 1.0e-4;
  const il::int_t max_nb_iterations = 10;
  il::Array<double> x{n, 0.0};
  il::Status status{};
  gmres.Solve(y.view(), relative_precision, max_nb_iterations, il::io, x.Edit(),
              status);
  status.AbortOnError();

  const il::int_t nb_iterations = gmres.nbIterations();

  return 0;
}
