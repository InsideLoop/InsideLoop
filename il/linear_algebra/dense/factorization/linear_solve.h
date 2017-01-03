//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_LINEAR_SOLVE_H
#define IL_LINEAR_SOLVE_H

#include <il/core/Status.h>

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/BandArray2C.h>
#include <il/TriDiagonal.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#endif

namespace il {

#ifdef IL_MKL
inline il::Array<double> linear_solve(il::Array2D<double> A,
                                      il::Array<double> y, il::io_t,
                                      il::Status &status) {
  IL_ASSERT(A.size(0) == A.size(1));
  IL_ASSERT(y.size() == A.size(0));

  il::Array<lapack_int> ipiv{A.size(0)};

  const int layout{LAPACK_COL_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size(0))};
  const lapack_int nrhs{1};
  const lapack_int lda{static_cast<lapack_int>(A.stride(1))};
  const lapack_int ldx{n};
  const lapack_int lapack_error{LAPACKE_dgesv(layout, n, nrhs, A.data(), lda,
                                              ipiv.data(), y.data(), ldx)};

  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    status.set(il::ErrorCode::ok);
  } else {
    status.set(il::ErrorCode::division_by_zero);
  }

  return y;
}

inline il::Array<double> linear_solve(il::Array2C<double> A,
                                      il::Array<double> y, il::io_t,
                                      il::Status &status) {
  IL_ASSERT(A.size(0) == A.size(1));
  IL_ASSERT(y.size() == A.size(1));

  il::Array<lapack_int> ipiv{A.size(0)};

  const int layout{LAPACK_ROW_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size(0))};
  const lapack_int nrhs{1};
  const lapack_int lda{static_cast<lapack_int>(A.stride(0))};
  const lapack_int ldx{1};
  const lapack_int lapack_error{LAPACKE_dgesv(layout, n, nrhs, A.data(), lda,
                                              ipiv.data(), y.data(), ldx)};

  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    status.set(il::ErrorCode::ok);
  } else {
    status.set(il::ErrorCode::division_by_zero);
  }

  return y;
}

inline il::Array<double> linear_solve(il::BandArray2C<double> A,
                                      il::Array<double> y, il::io_t,
                                      il::Status &status) {
  IL_ASSERT(A.size(0) == A.size(1));
  IL_ASSERT(y.size() == A.size(1));
  IL_ASSERT(A.capacity_right() >= A.width_left() + A.width_right());

  il::Array<lapack_int> ipiv{A.size(1)};

  const int layout{LAPACK_ROW_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size(1))};
  const lapack_int kl{static_cast<lapack_int>(A.width_left())};
  const lapack_int ku{static_cast<lapack_int>(A.width_right())};
  const lapack_int nrhs{1};
  const lapack_int ldab{
      static_cast<lapack_int>(A.width_left() + 1 + A.capacity_right())};
  const lapack_int ldb{1};
  const lapack_int lapack_error{LAPACKE_dgbsv(
      layout, n, kl, ku, nrhs, A.data(), ldab, ipiv.data(), y.data(), ldb)};

  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    status.set(il::ErrorCode::ok);
  } else {
    status.set(il::ErrorCode::division_by_zero);
  }

  return y;
}

inline il::Array<double> linear_solve(il::TriDiagonal<double> A,
                                      il::Array<double> y, il::io_t,
                                      il::Status &status) {
  IL_ASSERT(A.size() == y.size());

  const int layout{LAPACK_ROW_MAJOR};
  const lapack_int n{static_cast<lapack_int>(A.size())};
  const lapack_int nrhs{1};
  const lapack_int ldb{1};
  const lapack_int lapack_error{LAPACKE_dgtsv(layout, n, nrhs, A.data_lower(),
                                              A.data_diagonal(), A.data_upper(),
                                              y.data(), ldb)};

  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    status.set(il::ErrorCode::ok);
  } else {
    status.set(il::ErrorCode::division_by_zero);
  }

  return y;
}

#endif
}

#endif  // IL_LINEAR_SOLVE_H