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

#include <il/container/cuda/2d/CudaArray2D.h>

#include <cublas_v2.h>

namespace il {

inline void blas(float alpha, const il::CudaArray2D<float>& A,
                 const il::CudaArray2D<float>& B, float beta, il::io_t,
                 il::CudaArray2D<float>& C) {
  IL_ASSERT_PRECOND(A.size(1) == B.size(0));
  IL_ASSERT_PRECOND(A.size(0) == C.size(0));
  IL_ASSERT_PRECOND(B.size(1) == C.size(1));

  IL_ASSERT_PRECOND(A.size(1) == A.size(0));
  IL_ASSERT_PRECOND(B.size(1) == A.size(0));

  const int n = A.size(0);
  cublasHandle_t handle{};
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A.data(), n,
              B.data(), n, &beta, C.data(), n);
  cublasDestroy(handle);
}

inline void blas(double alpha, const il::CudaArray2D<double>& A,
                 const il::CudaArray2D<double>& B, double beta, il::io_t,
                 il::CudaArray2D<double>& C) {
  IL_ASSERT_PRECOND(A.size(1) == B.size(0));
  IL_ASSERT_PRECOND(A.size(0) == C.size(0));
  IL_ASSERT_PRECOND(B.size(1) == C.size(1));

  IL_ASSERT_PRECOND(A.size(1) == A.size(0));
  IL_ASSERT_PRECOND(B.size(1) == A.size(0));

  const int n = A.size(0);
  cublasHandle_t handle{};
  cublasCreate(&handle);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A.data(), n,
              B.data(), n, &beta, C.data(), n);
  cublasDestroy(handle);
}
}

#endif  // IL_CUDA_DOT_H
