//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUDA_BLAS_H
#define IL_CUDA_BLAS_H

#include <il/container/cuda/1d/CudaArray.h>
#include <il/container/cuda/2d/CudaArray2D.h>
#include <il/linear_algebra/cuda/dense/blas/CublasHandle.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 1
////////////////////////////////////////////////////////////////////////////////

inline void blas(float alpha, const il::CudaArray<float>& x, float beta,
                 il::io_t, il::CudaArray<float>& y, il::CublasHandle& handle) {
  IL_EXPECT_FAST(x.size() == y.size());

  const int n = static_cast<int>(x.size());
  const int incx = 1;
  const int incy = 1;
  cublasStatus_t status;
  if (beta == 1.0f) {
    status =
        cublasSaxpy(handle.handle(), n, &alpha, x.data(), incx, y.data(), incy);
    IL_EXPECT_FAST(status == CUBLAS_STATUS_SUCCESS);
  } else {
    const float beta_minus_one = beta - 1.0f;
    status = cublasSaxpy(handle.handle(), n, &beta_minus_one, y.data(), incx,
                         y.data(), incy);

    IL_EXPECT_FAST(status == CUBLAS_STATUS_SUCCESS);
    status =
        cublasSaxpy(handle.handle(), n, &alpha, x.data(), incx, y.data(), incy);
    IL_EXPECT_FAST(status == CUBLAS_STATUS_SUCCESS);
  }
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 3
////////////////////////////////////////////////////////////////////////////////

inline void blas(float alpha, const il::CudaArray2D<float>& A,
                 const il::CudaArray2D<float>& B, float beta, il::io_t,
                 il::CudaArray2D<float>& C, il::CublasHandle& handle) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(A.size(0) == C.size(0));
  IL_EXPECT_FAST(B.size(1) == C.size(1));

  IL_EXPECT_FAST(A.size(1) == A.size(0));
  IL_EXPECT_FAST(B.size(1) == A.size(0));

  const int n = A.size(0);
  cublasSgemm(handle.handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
              A.data(), n, B.data(), n, &beta, C.data(), n);
}

inline void blas(double alpha, const il::CudaArray2D<double>& A,
                 const il::CudaArray2D<double>& B, double beta, il::io_t,
                 il::CudaArray2D<double>& C, il::CublasHandle& handle) {
  IL_EXPECT_FAST(A.size(1) == B.size(0));
  IL_EXPECT_FAST(A.size(0) == C.size(0));
  IL_EXPECT_FAST(B.size(1) == C.size(1));

  IL_EXPECT_FAST(A.size(1) == A.size(0));
  IL_EXPECT_FAST(B.size(1) == A.size(0));

  const int n = A.size(0);
  cublasDgemm(handle.handle(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
              A.data(), n, B.data(), n, &beta, C.data(), n);
}
}

#endif  // IL_CUDA_BLAS_H
