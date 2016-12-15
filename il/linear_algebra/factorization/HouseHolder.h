//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HOUSEHOLDER_H
#define IL_HOUSEHOLDER_H

#include <il/container/1d/Array.h>
#include <il/container/2d/Array2D.h>
#include <il/core/Status.h>
#include <il/linear_algebra/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class HouseHolder {};

template <>
class HouseHolder<il::Array2D<double>> {
 private:
  il::Array<double> reflexion_;
  il::Array2D<double> house_holder_;

 public:
  // Computes a QR factorization of a general n0 x n1 matrix A
  //
  //  A = Q.R
  //
  HouseHolder(il::Array2D<double> A);

  // Size of the matrix
//  il::int_t size(il::int_t d) const;

  // Solve the system of equation with one second member
//  il::Array<double> solve(il::Array<double> y) const;
};

HouseHolder<il::Array2D<double>>::HouseHolder(il::Array2D<double> A)
    : house_holder_{} {
  IL_ASSERT_PRECOND(A.size(0) > 0);
  IL_ASSERT_PRECOND(A.size(1) > 0);

  const int layout = LAPACK_COL_MAJOR;
  const char trans = 'N';
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  il::Array<double> reflexion{m < n ? m : n};
  const lapack_int lapack_error =
      LAPACKE_dgeqrf(layout, m, n, A.data(), lda, reflexion.data());
  IL_ASSERT(lapack_error >= 0);
}

}

#endif  // IL_HOUSEHOLDER_H
