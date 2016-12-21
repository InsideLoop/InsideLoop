//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_BLAS_H
#define IL_BLAS_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>

#ifdef IL_MKL
#include <mkl_cblas.h>
#define IL_CBLAS_INT MKL_INT
#define IL_CBLAS_LAYOUT CBLAS_LAYOUT
#else
#include <cblas.h>
#define IL_CBLAS_INT int
#define IL_CBLAS_LAYOUT CBLAS_ORDER
#endif

namespace il {

enum class Blas {
  regular,
  transpose,
  conjugate_transpose,
  symmetric_upper,
  symmetric_lower,
  hermitian_upper,
  hermitian_lower
};

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 1
////////////////////////////////////////////////////////////////////////////////

// x, y are vectors
// y <- alpha x + y
inline void blas(double alpha, const il::Array<double>& x, il::io_t,
                 il::Array<double>& y) {
  IL_ASSERT(x.size() == y.size());

  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(x.size())};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};
  cblas_daxpy(n, alpha, x.data(), incx, y.data(), incy);
}

// x, y are vectors
// y <- alpha x + beta y
inline void blas(float alpha, const il::Array<float>& x, float beta,
                 il::io_t, il::Array<float>& y) {
  IL_ASSERT(x.size() == y.size());

  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(x.size())};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};
  cblas_saxpby(n, alpha, x.data(), incx, beta, y.data(), incy);
}

// x, y are vectors
// y <- alpha x + beta y
inline void blas(double alpha, const il::Array<double>& x, double beta,
                 il::io_t, il::Array<double>& y) {
  IL_ASSERT(x.size() == y.size());

  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(x.size())};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};
  cblas_daxpby(n, alpha, x.data(), incx, beta, y.data(), incy);
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 2
////////////////////////////////////////////////////////////////////////////////

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(float alpha, const il::Array2D<float>& A,
                 const il::Array<float>& x, float beta, il::io_t,
                 il::Array<float>& y) {
  IL_ASSERT(A.size(0) == y.size());
  IL_ASSERT(A.size(1) == x.size());
  IL_ASSERT(&x != &y);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};

  cblas_sgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.data(), incy);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(double alpha, const il::Array2D<double>& A,
                 const il::Array<double>& x, double beta, il::io_t,
                 il::Array<double>& y) {
  IL_ASSERT(A.size(0) == y.size());
  IL_ASSERT(A.size(1) == x.size());
  IL_ASSERT(&x != &y);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.data(), incy);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(double alpha, const il::Array2C<double>& A,
                 const il::Array<double>& x, double beta, il::io_t,
                 il::Array<double>& y) {
  IL_ASSERT(A.size(0) == y.size());
  IL_ASSERT(A.size(1) == x.size());
  IL_ASSERT(&x != &y);

  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.data(), incy);
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 3
////////////////////////////////////////////////////////////////////////////////

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(float alpha, const il::Array2D<float>& A,
                 const il::Array2D<float>& B, float beta, il::io_t,
                 il::Array2D<float>& C) {
  IL_ASSERT(A.size(1) == B.size(0));
  IL_ASSERT(C.size(0) == A.size(0));
  IL_ASSERT(C.size(1) == B.size(1));
  IL_ASSERT(&A != &C);
  IL_ASSERT(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(1))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(1))};
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.data(), ldc);
}

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(double alpha, const il::Array2D<double>& A,
                 const il::Array2D<double>& B, double beta, il::io_t,
                 il::Array2D<double>& C) {
  IL_ASSERT(A.size(1) == B.size(0));
  IL_ASSERT(C.size(0) == A.size(0));
  IL_ASSERT(C.size(1) == B.size(1));
  IL_ASSERT(&A != &C);
  IL_ASSERT(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(1))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(1))};
  cblas_dgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.data(), ldc);
}

inline void blas(double alpha, const il::Array2D<double>& A, Blas info_a,
                 const il::Array2D<double>& B, Blas info_b, double beta,
                 il::io_t, il::Array2D<double>& C) {
  IL_ASSERT(A.size(1) == B.size(0));
  IL_ASSERT(C.size(0) == A.size(0));
  IL_ASSERT(C.size(1) == B.size(1));
  IL_ASSERT(&A != &C);
  IL_ASSERT(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(1))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(1))};
  if (info_a == il::Blas::symmetric_upper && info_b == il::Blas::regular) {
    IL_ASSERT(A.size(0) == A.size(1));
    const CBLAS_SIDE side{CblasLeft};
    const CBLAS_UPLO uplo{CblasUpper};
    cblas_dsymm(layout, side, uplo, m, n, alpha, A.data(), lda, B.data(), ldb,
                beta, C.data(), ldc);
  } else {
    (void)transa;
    (void)transb;
    (void)k;
    abort();
  }
}

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(float alpha, const il::Array2C<float>& A,
                 const il::Array2C<float>& B, float beta, il::io_t,
                 il::Array2C<float>& C) {
  IL_ASSERT(A.size(1) == B.size(0));
  IL_ASSERT(C.size(0) == A.size(0));
  IL_ASSERT(C.size(1) == B.size(1));
  IL_ASSERT(&A != &C);
  IL_ASSERT(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(0))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(0))};
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.data(), ldc);
}

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(double alpha, const il::Array2C<double>& A,
                 const il::Array2C<double>& B, double beta, il::io_t,
                 il::Array2C<double>& C) {
  IL_ASSERT(A.size(1) == B.size(0));
  IL_ASSERT(C.size(0) == A.size(0));
  IL_ASSERT(C.size(1) == B.size(1));
  IL_ASSERT(&A != &C);
  IL_ASSERT(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(0))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(0))};
  cblas_dgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.data(), ldc);
}

inline void blas(double alpha, const il::Array2C<double>& A, Blas info_a,
                 const il::Array2C<double>& B, Blas info_b, double beta,
                 il::io_t, il::Array2C<double>& C) {
  IL_ASSERT(A.size(1) == B.size(0));
  IL_ASSERT(C.size(0) == A.size(0));
  IL_ASSERT(C.size(1) == B.size(1));
  IL_ASSERT(&A != &C);
  IL_ASSERT(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(0))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(0))};
  if (info_a == il::Blas::symmetric_upper && info_b == il::Blas::regular) {
    IL_ASSERT(A.size(0) == A.size(1));
    const CBLAS_SIDE side{CblasLeft};
    const CBLAS_UPLO uplo{CblasUpper};
    cblas_dsymm(layout, side, uplo, m, n, alpha, A.data(), lda, B.data(), ldb,
                beta, C.data(), ldc);
  } else {
    (void)transa;
    (void)transb;
    (void)k;
    abort();
  }
}
}

#endif  // IL_BLAS_H
