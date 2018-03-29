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

#ifndef IL_DOT_H
#define IL_DOT_H

#include <il/StaticArray.h>
#include <il/StaticArray2D.h>
#include <il/StaticArray3D.h>
#include <il/StaticArray4D.h>

#include <il/linearAlgebra/Matrix.h>
#include <il/linearAlgebra/dense/blas/blas.h>

namespace il {

inline float dot(const il::Array<float>& x, const il::Array<float>& y) {
  IL_EXPECT_FAST(x.size() == y.size());

  float sum = 0.0f;
  for (il::int_t i = 0; i < x.size(); ++i) {
    sum += x[i] * y[i];
  }

  return sum;
}

inline double dot(const il::Array<double>& x, const il::Array<double>& y) {
  IL_EXPECT_FAST(x.size() == y.size());

  double sum = 0.0;
  for (il::int_t i = 0; i < x.size(); ++i) {
    sum += x[i] * y[i];
  }

  return sum;
}

#ifdef IL_BLAS
inline il::Array<double> dot(const il::Array2D<double>& A,
                             const il::Array<double>& x) {
  IL_EXPECT_FAST(A.size(1) == x.size());

  il::Array<double> y{A.size(0)};
  il::blas(1.0, A, x, 0.0, il::io, y);

  return y;
}

inline il::Array<double> dot(const il::Array2C<double>& A,
                             const il::Array<double>& x) {
  IL_EXPECT_FAST(A.size(1) == x.size());

  il::Array<double> y{A.size(0)};
  il::blas(1.0, A, x, 0.0, il::io, y);

  return y;
}

inline il::Array2D<float> dot(const il::Array2D<float>& A,
                              const il::Array2D<float>& B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  il::Array2D<float> C{A.size(0), B.size(1)};
  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transpose = CblasNoTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(1));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_sgemm(layout, transpose, transpose, m, n, k, alpha, A.data(), lda,
              B.data(), ldb, beta, C.Data(), ldc);

  return C;
}

inline il::Array2D<double> dot(const il::Array2D<double>& A,
                               const il::Array2D<double>& B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  il::Array2D<double> C{A.size(0), B.size(1)};
  const IL_CBLAS_LAYOUT layout = CblasColMajor;
  const CBLAS_TRANSPOSE transpose = CblasNoTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(1));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const double alpha = 1.0;
  const double beta = 0.0;
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(1));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(1));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(1));
  cblas_dgemm(layout, transpose, transpose, m, n, k, alpha, A.data(), lda,
              B.data(), ldb, beta, C.Data(), ldc);

  return C;
}

inline il::Array2C<double> dot(const il::Array2C<double>& A,
                               const il::Array2C<double>& B) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));

  il::Array2C<double> C{A.size(0), B.size(1)};
  const IL_CBLAS_LAYOUT layout = CblasRowMajor;
  const CBLAS_TRANSPOSE transpose = CblasNoTrans;
  const IL_CBLAS_INT m = static_cast<IL_CBLAS_INT>(A.size(0));
  const IL_CBLAS_INT n = static_cast<IL_CBLAS_INT>(B.size(1));
  const IL_CBLAS_INT k = static_cast<IL_CBLAS_INT>(A.size(1));
  const double alpha = 1.0;
  const double beta = 0.0;
  const IL_CBLAS_INT lda = static_cast<IL_CBLAS_INT>(A.stride(0));
  const IL_CBLAS_INT ldb = static_cast<IL_CBLAS_INT>(B.stride(0));
  const IL_CBLAS_INT ldc = static_cast<IL_CBLAS_INT>(C.stride(0));
  cblas_dgemm(layout, transpose, transpose, m, n, k, alpha, A.data(), lda,
              B.data(), ldb, beta, C.Data(), ldc);

  return C;
}
#endif

template <typename T, il::int_t n>
T dot(const il::StaticArray<T, n>& x, const il::StaticArray<T, n>& y) {
  T ans{0};
  for (il::int_t i = 0; i < n; ++i) {
    ans += x[i] * y[i];
  }
  return ans;
}

template <typename T, il::int_t n0, il::int_t n>
il::StaticArray<T, n0> dot(const il::StaticArray2D<T, n0, n>& A,
                           const il::StaticArray<T, n>& B) {
  il::StaticArray<T, n0> C{0};
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      C[i0] += A(i0, i) * B[i];
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n>
il::StaticArray<T, n0> dot(const il::StaticArray2D<T, n, n0>& A,
                           il::MatrixOperator A_info,
                           const il::StaticArray<T, n>& B) {
  IL_EXPECT_FAST(A_info == il::MatrixOperator::Transpose);
  IL_UNUSED(A_info);

  il::StaticArray<T, n0> C{};
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    C[i0] = 0;
    for (il::int_t i = 0; i < n; ++i) {
      C[i0] += A(i, i0) * B[i];
    }
  }
  return C;
}

template <typename T, il::int_t n, il::int_t n1>
il::StaticArray<T, n1> dot(const il::StaticArray<T, n>& A,
                           const il::StaticArray2D<T, n, n1>& B) {
  il::StaticArray<T, n1> C{0};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i = 0; i < n; ++i) {
      C[i1] += A[i] * B(i, i1);
    }
  }

  return C;
}

template <typename T, il::int_t n0, il::int_t n, il::int_t n1>
il::StaticArray2D<T, n0, n1> dot(const il::StaticArray2D<T, n0, n>& A,
                                 const il::StaticArray2D<T, n, n1>& B) {
  il::StaticArray2D<T, n0, n1> C{0};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i = 0; i < n; ++i) {
      for (il::int_t i0 = 0; i0 < n0; ++i0) {
        C(i0, i1) += A(i0, i) * B(i, i1);
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n, il::int_t n1>
il::StaticArray2D<T, n0, n1> dot(const il::StaticArray2D<T, n, n0>& A,
                                 il::MatrixOperator A_info,
                                 const il::StaticArray2D<T, n, n1>& B) {
  IL_EXPECT_FAST(A_info == il::MatrixOperator::Transpose);
  IL_UNUSED(A_info);

  il::StaticArray2D<T, n0, n1> C{};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      C(i0, i1) = 0;
      for (il::int_t i = 0; i < n; ++i) {
        C(i0, i1) += A(i, i0) * B(i, i1);
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n, il::int_t n1, il::int_t n2>
il::StaticArray3D<T, n0, n1, n2> dot(const il::StaticArray2D<T, n0, n>& A,
                                     const il::StaticArray3D<T, n, n1, n2>& B) {
  il::StaticArray3D<T, n0, n1, n2> C{0};
  for (il::int_t i = 0; i < n; ++i) {
    for (il::int_t i2 = 0; i2 < n2; ++i2) {
      for (il::int_t i1 = 0; i1 < n1; ++i1) {
        for (il::int_t i0 = 0; i0 < n0; ++i0) {
          C(i0, i1, i2) += A(i0, i) * B(i, i1, i2);
        }
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n>
il::StaticArray2D<T, n0, n1> dot(const il::StaticArray3D<T, n0, n1, n>& A,
                                 const il::StaticArray<T, n>& B) {
  il::StaticArray2D<T, n0, n1> C{};
  for (il::int_t i1 = 0; i1 < n1; ++i1) {
    for (il::int_t i0 = 0; i0 < n0; ++i0) {
      C(i0, i1) = 0;
      for (il::int_t i = 0; i < n; ++i) {
        C(i0, i1) += A(i0, i1, i) * B[i];
      }
    }
  }
  return C;
}

template <typename T, il::int_t n0, il::int_t n1, il::int_t n2, il::int_t n>
il::StaticArray3D<T, n0, n1, n2> dot(
    const il::StaticArray4D<T, n0, n1, n2, n>& A,
    const il::StaticArray<T, n>& B) {
  il::StaticArray3D<T, n0, n1, n2> C{};
  for (il::int_t i0 = 0; i0 < n0; ++i0) {
    for (il::int_t i1 = 0; i1 < n1; ++i1) {
      for (il::int_t i2 = 0; i2 < n2; ++i2) {
        C(i0, i1, i2) = 0.0;
        for (il::int_t i = 0; i < n; ++i) {
          C(i0, i1, i2) += A(i0, i1, i2, i) * B[i];
        }
      }
    }
  }
  return C;
}
}  // namespace il

#endif  // IL_DOT_H
