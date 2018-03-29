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

#ifndef IL_GMRES_H
#define IL_GMRES_H

#include <il/ArrayView.h>
#include <il/StaticArray.h>

#include "mkl_blas.h"
#include "mkl_rci.h"

namespace il {

template <typename T>
class ArrayFunctor {
 public:
  virtual il::int_t sizeInput() const = 0;
  virtual il::int_t sizeOutput() const = 0;
  virtual void operator()(il::ArrayView<T> x, il::io_t,
                          il::ArrayEdit<T> y) const = 0;
};

class Gmres {
 private:
  double relative_precision_;
  il::int_t max_nb_iterations_;
  il::int_t restart_iteration_;
  il::int_t nb_iterations_;

 public:
  Gmres();
  Gmres(double relative_precision, il::int_t max_nb_iterations,
        il::int_t restart_iteration);
  void Solve(const ArrayFunctor<double>& a, const ArrayFunctor<double>& b,
             il::ArrayView<double> y, bool use_preconditionner,
             bool use_x_as_initial_value, il::io_t, il::ArrayEdit<double> x);
  il::int_t nbIterations() const;
};

}  // namespace il

#endif  // IL_GMRES_H
