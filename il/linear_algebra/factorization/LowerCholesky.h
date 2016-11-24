//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_LOWERCHOLESKY_H
#define IL_LOWERCHOLESKY_H

#include <il/core/Error.h>
#include <il/container/2d/LowerArray2D.h>

#include <iostream>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

namespace il {

class LowerCholesky {
 private:
  il::LowerArray2D<double> l_;

 public:
  LowerCholesky(il::LowerArray2D<double> A, il::io_t, il::Error& error);
};

LowerCholesky::LowerCholesky(il::LowerArray2D<double> A, il::io_t,
                             il::Error& error)
    : l_{} {
  const int layout{LAPACK_COL_MAJOR};
  const char uplo{'L'};
  const lapack_int n{static_cast<lapack_int>(A.size())};
  const lapack_int lapack_error{LAPACKE_dpptrf(layout, uplo, n, A.data())};
  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    error.set(ErrorCode::ok);
    l_ = std::move(A);
  } else {
    error.set(ErrorCode::negative_number);
  }
}
}

#endif  // IL_LOWERCHOLESKY_H
