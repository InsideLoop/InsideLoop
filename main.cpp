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
  il::Array<double> diag{il::value, {1.0, 2.0, 3.0}};
  Matrix<double> m{diag};
  il::Array<double> y{il::value, {1.0, 1.0, 1.0}};

  // Conjugate gradient solver
  {
    il::print("Conjugate gradient solver\n");
    il::Array<double> precond_diag{il::value, {1.0, 0.5, 0.3}};
    Matrix<double> precond{precond_diag};

    {
      il::Cg<double> cg{m, precond};
      cg.SetRelativePrecision(1.0e-1);
      cg.SetAbsolutionPrecision(0.0);
      cg.SetMaxNbIterations(2);

      il::Status status{};
      il::Array<double> x = cg.Solve(y, il::io, status);
      status.AbortOnError();

      il::print("Number of iterations: {}\n", cg.nbIterations());
      il::print("Norm of the residual: {}\n", cg.normResidual());
      il::print("\n");
    }
    {
      il::Array<double> x{3};

      il::Cg<double> cg{m, precond};
      cg.SetToSolve(y.view());
      for (il::int_t i = 0; i < 10; ++i) {
        cg.getSolution(il::io, x.Edit());
        il::print("Iteration: {:2d}  --", cg.nbIterations());
        il::print("  x = {:8.6f} - {:8.6f} - {:8.6f}  --", x[0], x[1], x[2]);
        il::print("  residual = {:8.2e}\n", cg.normResidual());
        cg.Next();
      }
    }
  }

  // GMRES solver
  il::print("\nGMRES solver\n");
  il::Array<double> diag_precond{il::value, {1.0, 1.0, 1.0}};
  Matrix<double> precond{diag_precond};
  il::Array<double> x{3};

  il::int_t krylov_dim = 20;
  il::Gmres<double> gmres{m, precond, krylov_dim};
  gmres.SetToSolve(y);
  for (il::int_t i = 0; i < 10; ++i) {
    gmres.getSolution(il::io, x);
    il::print("Iteration: {:2d}  --", gmres.nbIterations());
    il::print("  x = {:8.6f} - {:8.6f} - {:8.6f}  --", x[0], x[1], x[2]);
    il::print("  residual = {:8.2e}\n", gmres.normResidual());
    gmres.Next();
  }

  return 0;
}
