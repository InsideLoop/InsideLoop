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

#include <gtest/gtest.h>

#include <il/linear_algebra/dense/blas/blas.h>

#ifdef IL_BLAS
TEST(Blas, daxpy) {
  il::Array<double> x{il::value, {1.0, 2.0}};
  il::Array<double> y{il::value, {1.0, 1.0}};

  // y <- 2.0 * x + y
  il::blas(2.0, x, il::io, y);

  ASSERT_TRUE(y.size() == 2 && y[0] == 3.0 && y[1] == 5.0);
}
#endif
