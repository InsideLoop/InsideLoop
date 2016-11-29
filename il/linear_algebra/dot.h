//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_DOT_H
#define IL_DOT_H

#include <il/linear_algebra/blas.h>

namespace il {

inline il::Array<double> dot(const il::Array2D<double>& A,
                             const il::Array<double>& x) {
  IL_ASSERT(A.size(1) == x.size());

  il::Array<double> y{A.size(0)};
  il::blas(1.0, A, x, 0.0, il::io, y);

  return y;
}

inline il::Array<double> dot(const il::Array2C<double>& A,
                             const il::Array<double>& x) {
  IL_ASSERT(A.size(1) == x.size());

  il::Array<double> y{A.size(0)};
  il::blas(1.0, A, x, 0.0, il::io, y);

  return y;
}

inline il::Array2D<double> dot(const il::Array2D<double>& A,
                               const il::Array2D<double>& B) {
  IL_ASSERT(A.size(1) == B.size(0));

  il::Array2D<double> C{A.size(0), B.size(1)};
  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transpose{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const double alpha{1.0};
  const double beta{0.0};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(1))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(1))};
  cblas_dgemm(layout, transpose, transpose, m, n, k, alpha, A.data(), lda,
              B.data(), ldb, beta, C.data(), ldc);

  return C;
}

inline il::Array2C<double> dot(const il::Array2C<double>& A,
                               const il::Array2C<double>& B) {
  IL_ASSERT(A.size(1) == B.size(0));

  il::Array2C<double> C{A.size(0), B.size(1)};
  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transpose{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const double alpha{1.0};
  const double beta{0.0};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(0))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(0))};
  cblas_dgemm(layout, transpose, transpose, m, n, k, alpha, A.data(), lda,
              B.data(), ldb, beta, C.data(), ldc);

  return C;
}
}

#endif  // IL_DOT_H
