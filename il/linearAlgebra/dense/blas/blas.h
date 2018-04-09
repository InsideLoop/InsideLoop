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

#ifndef IL_BLAS_H
#define IL_BLAS_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/StaticArray.h>
#include <il/StaticArray2D.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>
#include <il/linearAlgebra/Matrix.h>

#ifdef IL_MKL
#include <mkl_cblas.h>
#define IL_CBLAS_INT MKL_INT
#define IL_CBLAS_LAYOUT CBLAS_LAYOUT
#elif IL_OPENBLAS
#include <OpenBLAS/cblas.h>
#define IL_CBLAS_INT int
#define IL_CBLAS_LAYOUT CBLAS_ORDER
#endif

namespace il {

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 1
////////////////////////////////////////////////////////////////////////////////

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2>
void blas(T alpha, const il::StaticArray3D<T, n0, n1, n2> &A, T beta, il::io_t,
          il::StaticArray3D<T, n0, n1, n2> &B) {
  for (il::int_t i2 = 0; i2 < n2; ++i2) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        B(i0, i1, i2) = alpha * A(i0, i1, i2) + beta * B(i0, i1, i2);
      }
    }
  }
}

#ifdef IL_BLAS
// x, y are vectors
// y <- alpha x + y
inline void blas(double alpha, const il::Array<double> &x, il::io_t,
                 il::Array<double> &y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(x.size())};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};
  cblas_daxpy(n, alpha, x.data(), incx, y.Data(), incy);
}

// x, y are vectors
// y <- alpha x + beta y
inline void blas(float alpha, const il::Array<float> &x, float beta, il::io_t,
                 il::Array<float> &y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(x.size())};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};
  cblas_saxpby(n, alpha, x.data(), incx, beta, y.Data(), incy);
}

// x, y are vectors
// y <- alpha x + beta y
inline void blas(double alpha, const il::Array<double> &x, double beta,
                 il::io_t, il::Array<double> &y) {
  IL_EXPECT_FAST(x.size() == y.size());

  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(x.size())};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};
  cblas_daxpby(n, alpha, x.data(), incx, beta, y.Data(), incy);
}

inline void blas(double alpha, const il::Array2D<double> &a, double beta,
                 il::io_t, il::Array2D<double> &b) {
  IL_EXPECT_FAST(a.size(0) == b.size(0));
  IL_EXPECT_FAST(a.size(1) == b.size(1));

  const il::int_t n0 = a.size(0);
  const il::int_t n1 = a.size(1);
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      b(i0, i1) = beta * b(i0, i1) + alpha * a(i0, i1);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 2
////////////////////////////////////////////////////////////////////////////////

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(float alpha, const il::Array2D<float> &A,
                 const il::Array<float> &x, float beta, il::io_t,
                 il::Array<float> &y) {
  IL_EXPECT_FAST(A.size(0) == y.size());
  IL_EXPECT_FAST(A.size(1) == x.size());
  IL_EXPECT_FAST(&x != &y);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};

  cblas_sgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(double alpha, const il::Array2D<double> &A,
                 const il::Array<double> &x, double beta, il::io_t,
                 il::Array<double> &y) {
  IL_EXPECT_FAST(A.size(0) == y.size());
  IL_EXPECT_FAST(A.size(1) == x.size());
  IL_EXPECT_FAST(&x != &y);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(double alpha, const il::Array2DView<double> &A,
                 const il::ArrayView<double> &x, double beta, il::io_t,
                 il::ArrayEdit<double> y) {
  IL_EXPECT_FAST(A.size(0) == y.size());
  IL_EXPECT_FAST(A.size(1) == x.size());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(double alpha, const il::Array2DView<double> &A,
                 il::Dot op, const il::ArrayView<double> &x,
                 double beta, il::io_t, il::ArrayEdit<double> y) {
  IL_EXPECT_FAST(op == il::Dot::Transpose);
  IL_EXPECT_FAST(A.size(1) == y.size());
  IL_EXPECT_FAST(A.size(0) == x.size());

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = CblasTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT incx = 1;
  const IL_CBLAS_INT incy = 1;

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
inline void blas(double alpha, const il::Array2C<double> &A,
                 const il::Array<double> &x, double beta, il::io_t,
                 il::Array<double> &y) {
  IL_EXPECT_FAST(A.size(0) == y.size());
  IL_EXPECT_FAST(A.size(1) == x.size());
  IL_EXPECT_FAST(&x != &y);

  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT incx{1};
  const IL_CBLAS_INT incy{1};

  cblas_dgemv(layout, transa, m, n, alpha, A.data(), lda, x.data(), incx, beta,
              y.Data(), incy);
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 3
////////////////////////////////////////////////////////////////////////////////

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(float alpha, const il::Array2D<float> &A,
                 const il::Array2D<float> &B, float beta, il::io_t,
                 il::Array2D<float> &C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

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
              ldb, beta, C.Data(), ldc);
}

inline void blas(float alpha, const il::Array2DView<float> &A,
                 const il::Array2DView<float> &B, il::Dot op,
                 float beta, il::io_t, il::Array2DEdit<float> C) {
  IL_EXPECT_FAST(A.size(1) == B.size(1));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(0));
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_TRANSPOSE transb = CblasTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

// A is a matrix, x, y are vectors
// y <- alpha A.x + beta y
template <il::int_t n>
void blas(double alpha, const il::StaticArray2D<double, n, n> &A,
          const il::StaticArray2D<double, n, n> &B, double beta, il::io_t,
          il::StaticArray2D<double, n, n> &C) {
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_TRANSPOSE transb = CblasNoTrans;
  const IL_CBLAS_INT lapack_n = static_cast<IL_CBLAS_INT>(n);

  cblas_dgemm(layout, transa, transb, lapack_n, lapack_n, lapack_n, alpha,
              A.data(), lapack_n, B.data(), lapack_n, beta, C.Data(), lapack_n);
}

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(double alpha, const il::Array2D<double> &A,
                 const il::Array2D<double> &B, double beta, il::io_t,
                 il::Array2D<double> &C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

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
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, const il::Array2DView<double> &A,
                 const il::Array2DView<double> &B, double beta, il::io_t,
                 il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE trans = CblasNoTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(1));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_dgemm(layout, trans, trans, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, const il::Array2DView<double> &A,
                 il::Dot op, const il::Array2DView<double> &B,
                 double beta, il::io_t, il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(A.size(0) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(1));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = CblasTrans;
  const CBLAS_TRANSPOSE transb = CblasNoTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(1));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_dgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, const il::Array2DView<double> &A,
                 const il::Array2DView<double> &B, il::Dot op,
                 double beta, il::io_t, il::Array2DEdit<double> C) {
  IL_EXPECT_FAST(A.size(1) == B.size(1));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(0));
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_TRANSPOSE transb = CblasTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_dgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, const il::Array2D<double> &A,
                 il::MatrixType info_a, const il::Array2D<double> &B,
                 il::MatrixType info_b, double beta, il::io_t,
                 il::Array2D<double> &C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasColMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(1))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(1))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(1))};
  if (info_a == il::MatrixType::SymmetricUpper &&
      info_b == il::MatrixType::Regular) {
    IL_EXPECT_FAST(A.size(0) == A.size(1));
    const CBLAS_SIDE side{CblasLeft};
    const CBLAS_UPLO uplo{CblasUpper};
    cblas_dsymm(layout, side, uplo, m, n, alpha, A.data(), lda, B.data(), ldb,
                beta, C.Data(), ldc);
  } else {
    (void)transa;
    (void)transb;
    (void)k;
    abort();
  }
}

// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(float alpha, const il::Array2C<float> &A,
                 const il::Array2C<float> &B, float beta, il::io_t,
                 il::Array2C<float> &C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

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
              ldb, beta, C.Data(), ldc);
}

inline void blas(float alpha, const il::Array2CView<float> &A,
                 const il::Array2CView<float> &B, il::Dot op,
                 float beta, il::io_t, il::Array2CEdit<float> C) {
  IL_EXPECT_FAST(A.size(1) == B.size(1));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(0));
  IL_EXPECT_FAST(op == il::Dot::Transpose);

  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transa = CblasNoTrans;
  const CBLAS_TRANSPOSE transb = CblasTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(0));
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_sgemm(layout, transa, transb, m, n, k, alpha, A.data(), lda, B.data(),
              ldb, beta, C.Data(), ldc);
}


// A, B, C are matrices
// C <- alpha A.B + beta C
inline void blas(double alpha, const il::Array2C<double> &A,
                 const il::Array2C<double> &B, double beta, il::io_t,
                 il::Array2C<double> &C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

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
              ldb, beta, C.Data(), ldc);
}

inline void blas(double alpha, const il::Array2C<double> &A,
                 il::MatrixType info_a, const il::Array2C<double> &B,
                 il::MatrixType info_b, double beta, il::io_t,
                 il::Array2C<double> &C) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(C.size(0) == A.size(0));
  IL_EXPECT_FAST(C.size(1) == B.size(1));
  IL_EXPECT_FAST(&A != &C);
  IL_EXPECT_FAST(&B != &C);

  const IL_CBLAS_LAYOUT layout{CblasRowMajor};
  const CBLAS_TRANSPOSE transa{CblasNoTrans};
  const CBLAS_TRANSPOSE transb{CblasNoTrans};
  const IL_CBLAS_INT m{static_cast<IL_CBLAS_INT>(A.size(0))};
  const IL_CBLAS_INT n{static_cast<IL_CBLAS_INT>(B.size(1))};
  const IL_CBLAS_INT k{static_cast<IL_CBLAS_INT>(A.size(1))};
  const IL_CBLAS_INT lda{static_cast<IL_CBLAS_INT>(A.stride(0))};
  const IL_CBLAS_INT ldb{static_cast<IL_CBLAS_INT>(B.stride(0))};
  const IL_CBLAS_INT ldc{static_cast<IL_CBLAS_INT>(C.stride(0))};
  if (info_a == il::MatrixType::SymmetricUpper &&
      info_b == il::MatrixType::Regular) {
    IL_EXPECT_FAST(A.size(0) == A.size(1));
    const CBLAS_SIDE side{CblasLeft};
    const CBLAS_UPLO uplo{CblasUpper};
    cblas_dsymm(layout, side, uplo, m, n, alpha, A.data(), lda, B.data(), ldb,
                beta, C.Data(), ldc);
  } else {
    (void)transa;
    (void)transb;
    (void)k;
    abort();
  }
}
#endif  // IL_MKL

template <typename T, il::int_t n0, il::int_t n>
void blas(double alpha, const il::StaticArray2D<T, n0, n> &A,
          const il::StaticArray<T, n> &B, double beta, il::io_t,
          il::StaticArray<T, n0> &C) {
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    C[i0] *= beta;
    for (il::int_t i = 0; i < n; ++i) {
      C[i0] += alpha * A(i0, i) * B[i];
    }
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n>
void blas(double alpha, const il::StaticArray3D<T, n0, n1, n> &A,
          const il::StaticArray<T, n> &B, double beta, il::io_t,
          il::StaticArray2D<T, n0, n1> &C) {
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      C(i0, i1) *= beta;
      for (il::int_t i = 0; i < n; ++i) {
        C(i0, i1) += alpha * A(i0, i1, i) * B[i];
      }
    }
  }
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n>
void blas(double alpha, const il::StaticArray4D<T, n0, n1, n2, n> &A,
          const il::StaticArray<T, n> &B, double beta, il::io_t,
          il::StaticArray3D<T, n0, n1, n2> &C) {
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        C(i0, i1, i2) *= beta;
        for (il::int_t i = 0; i < n; ++i) {
          C(i0, i1, i2) += alpha * A(i0, i1, i2, i) * B[i];
        }
      }
    }
  }
}
}  // namespace il

#endif  // IL_BLAS_H
