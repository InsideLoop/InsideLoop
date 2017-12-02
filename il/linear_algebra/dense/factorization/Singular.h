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

#ifndef IL_SINGULAR_H
#define IL_SINGULAR_H

#include <il/Status.h>
#include <il/container/1d/Array.h>
#include <il/container/2d/Array2D.h>
#include <il/linear_algebra/dense/blas/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class Singular {};

template <>
class Singular<il::Array2D<double>> {
 private:
  il::Array<double> singular_value_;

 public:
  // Computes singular values of A
  Singular(il::Array2D<double> A, il::io_t, il::Status& status);
};

Singular<il::Array2D<double>>::Singular(il::Array2D<double> A, il::io_t,
                                        il::Status& status)
    : singular_value_{} {
  IL_EXPECT_FAST(A.size(0) > 0);
  IL_EXPECT_FAST(A.size(1) > 0);
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  const il::int_t min_mn = m < n ? m : n;
  il::Array<double> d{min_mn};
  il::Array<double> e{(min_mn == 1) ? 1 : (min_mn - 1)};
  il::Array<double> tauq{min_mn};
  il::Array<double> taup{min_mn};
  lapack_int lapack_error =
      LAPACKE_dgebrd(layout, m, n, A.data(), lda, d.data(), e.data(),
                     tauq.data(), taup.data());
  IL_EXPECT_FAST(lapack_error >= 0);

  const char uplo = (m >= n) ? 'U' : 'L';
  const lapack_int ncvt = 0;
  const lapack_int ldvt = 1;
  const lapack_int nru = 0;
  const lapack_int ldu = 1;
  const lapack_int ncc = 0;  // No matrix C is upplied
  const lapack_int ldc = 1;  // No matrix C is upplied
  il::Array<double> vt{ldvt * ncvt};
  il::Array<double> u{ldu * n};
  il::Array<double> c{1};  // Should be useless
  lapack_error =
      LAPACKE_dbdsqr(layout, uplo, n, ncvt, nru, ncc, d.data(), e.data(),
                     vt.data(), ldvt, u.data(), ldu, c.data(), ldc);

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.set(ErrorCode::ok);
    singular_value_ = std::move(d);
  } else {
    status.set(ErrorCode::no_convergence);
  }
}
}  // namespace il

#endif  // IL_SINGULAR_H
