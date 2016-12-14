//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_PARTIALLU_H
#define IL_PARTIALLU_H

#include <il/container/1d/Array.h>
#include <il/container/2d/Array2C.h>
#include <il/container/2d/Array2D.h>
#include <il/container/2d/LowerArray2D.h>
#include <il/container/2d/UpperArray2D.h>
#include <il/core/Error.h>
#include <il/linear_algebra/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class PartialLU {};

template <>
class PartialLU<il::Array2D<double>> {
 private:
  il::Array<lapack_int> ipiv_;
  il::Array2D<double> lu_;

 public:
  // Computes a LU factorization of a genral n0 x n1 matrix A using partial
  // pivoting with row interchanges. The factorization has the form
  //
  //  A = P.L.U
  //
  // where P is a permutation matrix, L is lower triangular with unit diagonal
  // elements, and U is upper triangular.
  PartialLU(il::Array2D<double> A, il::io_t, il::Error& error);

  // Size of the matrix
  il::int_t size(il::int_t d) const;

  // Read access to the L part of the decomposition
  const double& L(il::int_t i, il::int_t j) const;

  // Read access to the U part of the decomposition
  const double& U(il::int_t i, il::int_t j) const;

  // Solve the system of equation with one second member
  il::Array<double> solve(il::Array<double> y) const;

  // Solve the system of equation with many second member
  il::Array2D<double> solve(il::Array2D<double> y) const;

  // Compute the inverse of the matrix
  il::Array2D<double> inverse() const;

  // Compute an approximation of the condition number
  double condition_number(il::Norm norm_type, double norm_a) const;

  // Get the L part of the matrix
  il::LowerArray2D<double> L() const;

  // Get the U part of the matrix
  il::UpperArray2D<double> U() const;
};

PartialLU<il::Array2D<double>>::PartialLU(il::Array2D<double> A, il::io_t,
                                          il::Error& error)
    : ipiv_{}, lu_{} {
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int m = static_cast<lapack_int>(A.size(0));
  const lapack_int n = static_cast<lapack_int>(A.size(1));
  const lapack_int lda = static_cast<lapack_int>(A.stride(1));
  il::Array<lapack_int> ipiv{A.size(0) < A.size(1) ? A.size(0) : A.size(1)};
  const lapack_int lapack_error =
      LAPACKE_dgetrf(layout, m, n, A.data(), lda, ipiv.data());

  IL_ASSERT(lapack_error >= 0);
  if (lapack_error == 0) {
    error.set(ErrorCode::ok);
    ipiv_ = std::move(ipiv);
    lu_ = std::move(A);
  } else {
    error.set(ErrorCode::division_by_zero);
  }
}

il::int_t PartialLU<il::Array2D<double>>::size(il::int_t d) const {
  IL_ASSERT_BOUNDS(static_cast<il::uint_t>(d) < static_cast<il::uint_t>(2));
  return lu_.size(d);
}

il::Array<double> PartialLU<il::Array2D<double>>::solve(
    il::Array<double> y) const {
  IL_ASSERT_PRECOND(lu_.size(0) == lu_.size(1));

  const int layout = LAPACK_COL_MAJOR;
  const char trans = 'N';
  const lapack_int n = static_cast<lapack_int>(lu_.size(0));
  const lapack_int nrhs = 1;
  const lapack_int lda = static_cast<lapack_int>(lu_.stride(1));
  const lapack_int ldy = n;
  const lapack_int lapack_error = LAPACKE_dgetrs(
      layout, trans, n, nrhs, lu_.data(), lda, ipiv_.data(), y.data(), ldy);
  IL_ASSERT(lapack_error == 0);

  return y;
}

il::Array2D<double> PartialLU<il::Array2D<double>>::solve(
    il::Array2D<double> y) const {
  IL_ASSERT_PRECOND(lu_.size(0) == lu_.size(1));
  IL_ASSERT_PRECOND(lu_.size(0) == y.size(0));

  const int layout = LAPACK_COL_MAJOR;
  const char trans = 'N';
  const lapack_int n = static_cast<lapack_int>(lu_.size(0));
  const lapack_int nrhs = static_cast<lapack_int>(y.size(1));
  const lapack_int lda = static_cast<lapack_int>(lu_.stride(1));
  const lapack_int ldy = n;
  const lapack_int lapack_error = LAPACKE_dgetrs(
      layout, trans, n, nrhs, lu_.data(), lda, ipiv_.data(), y.data(), ldy);
  IL_ASSERT(lapack_error == 0);

  return y;
}

il::Array2D<double> PartialLU<il::Array2D<double>>::inverse() const {
  IL_ASSERT_PRECOND(lu_.size(0) == lu_.size(1));

  il::Array2D<double> inverse{lu_};
  const int layout = LAPACK_COL_MAJOR;
  const lapack_int n = static_cast<lapack_int>(inverse.size(0));
  const lapack_int lda = static_cast<lapack_int>(inverse.stride(1));
  const lapack_int lapack_error =
      LAPACKE_dgetri(layout, n, inverse.data(), lda, ipiv_.data());
  IL_ASSERT(lapack_error == 0);

  return inverse;
}

double PartialLU<il::Array2D<double>>::condition_number(il::Norm norm_type,
                                                        double norm_a) const {
  IL_ASSERT_PRECOND(lu_.size(0) == lu_.size(1));
  IL_ASSERT_PRECOND(norm_type == il::Norm::L1 || norm_type == il::Norm::Linf);

  const int layout = LAPACK_COL_MAJOR;
  const char lapack_norm = (norm_type == il::Norm::L1) ? '1' : 'I';
  const lapack_int n = static_cast<lapack_int>(lu_.size(0));
  const lapack_int lda = static_cast<lapack_int>(lu_.stride(1));
  double rcond;
  const lapack_int lapack_error =
      LAPACKE_dgecon(layout, lapack_norm, n, lu_.data(), lda, norm_a, &rcond);
  IL_ASSERT(lapack_error == 0);

  return 1.0 / rcond;
}

const double& PartialLU<il::Array2D<double>>::L(il::int_t i,
                                                il::int_t j) const {
  IL_ASSERT_BOUNDS(j < i);
  return lu_(i, j);
}

const double& PartialLU<il::Array2D<double>>::U(il::int_t i,
                                                il::int_t j) const {
  IL_ASSERT_BOUNDS(j >= i);
  return lu_(i, j);
}

il::LowerArray2D<double> PartialLU<il::Array2D<double>>::L() const {
  IL_ASSERT_PRECOND(size(0) == size(1));

  const il::int_t n = size(0);
  il::LowerArray2D<double> L{n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    L(i1, i1) = 1.0;
    for (il::int_t i0 = i1 + 1; i0 < n; ++i0) {
      L(i0, i1) = lu_(i0, i1);
    }
  }
  return L;
}

il::UpperArray2D<double> PartialLU<il::Array2D<double>>::U() const {
  IL_ASSERT_PRECOND(size(0) == size(1));

  const il::int_t n = size(0);
  il::UpperArray2D<double> U{n};
  for (il::int_t i1 = 0; i1 < n; ++i1) {
    for (il::int_t i0 = 0; i0 <= i1; ++i0) {
      U(i0, i1) = lu_(i0, i1);
    }
  }
  return U;
}
}

#endif  // IL_PARTIALLU_H
