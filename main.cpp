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
#include <il/Cg.h>
#include <il/Dynamic.h>
#include <il/Gmres.h>
#include <il/Map.h>
#include <il/print.h>

template <typename T>
class Matrix : public il::FunctorArray<T> {
 private:
  il::int_t n_;

 public:
  Matrix(il::int_t n) {
    n_ = n;
  }
  il::int_t size(il::int_t d) const override {
    IL_EXPECT_MEDIUM(d == 0 || d == 1);

    return n_;
  }
  void operator()(il::ArrayView<T> x, il::io_t,
                  il::ArrayEdit<T> y) const override {
    IL_EXPECT_FAST(x.size() == n_);
    IL_EXPECT_FAST(y.size() == n_);

    double h = 1.0 / (n_ + 1);
    y[0] = h * h * (- x[1] + 2 * x[0]);
    for (il::int_t i = 1; i < n_ - 1; ++i) {
      y[i] = h * h * (- x[i + 1] + 2 * x[i] - x[i - 1]);
    }
    y[n_ - 1] = h * h * (2 * x[n_ - 1] - x[n_ - 2]);
  }
};

int main() {
  il::int_t n = 10;
  Matrix<double> A{n};

  il::Array<double> y{n};
  double h = 1.0 / (n + 1);
  for (il::int_t i = 0; i < n; ++i) {
    y[i] = il::pi * il::pi * std::sin(il::pi * il::ipow<2>((i + 1) * h));
  }

  il::Cg<double> cg{A};
  cg.SetRelativePrecision(1.0e-6);
  cg.SetMaxNbIterations(100);

  il::Status status{};
  il::Array<double> x = cg.Solve(y, il::io, status);
  status.AbortOnError();

  il::print("Number of iterations: {}\n", cg.nbIterations());
  il::print("True residual norm: {}\n", cg.trueResidualNorm());
  il::print("\n");

  return 0;
}
