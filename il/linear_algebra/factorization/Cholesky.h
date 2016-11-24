//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CHOLESKY_H
#define IL_CHOLESKY_H

#include <il/core/Error.h>
#include <il/container/2d/Array2D.h>

#include <iostream>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

namespace il {

class Cholesky {
 private:
  il::Array2D<double> l_;

 public:
  Cholesky(il::Array2D<double> A, il::io_t, il::Error& error);
};

Cholesky::Cholesky(il::Array2D<double> A, il::io_t, il::Error& error) : l_{} {
  IL_ASSERT(A.size(0) == A.size(1));

  const int layout{LAPACK_COL_MAJOR};
  const char uplo{'L'};
  const lapack_int n{static_cast<lapack_int>(A.size(0))};
  const lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  const lapack_int lapack_error{LAPACKE_dpotrf(layout, uplo, n, A.data(), lda)};
  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    error.set(ErrorCode::ok);
    l_ = std::move(A);
  } else {
    error.set(ErrorCode::negative_number);
  }
}
}

#endif  // IL_CHOLESKY_H
