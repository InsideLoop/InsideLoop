//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUDA_DOT_H
#define IL_CUDA_DOT_H

#include <il/linear_algebra/cuda/dense/blas/CublasHandle.h>
#include <il/container/cuda/1d/CudaArray.h>
#include <il/container/cuda/2d/CudaArray2D.h>

#include <cublas_v2.h>

namespace il {

inline float dot(const il::CudaArray<float>& x, const il::CudaArray<float>& y,
                 il::io_t, il::CublasHandle& handle) {
  IL_EXPECT_FAST(x.size() == y.size());

  const int n = x.size();
  const int incx = 1;
  const int incy = 1;
  float ans;
  cublasStatus_t status =
      cublasSdot(handle.handle(), n, x.data(), incx, y.data(), incy, &ans);

  return ans;
}

}

#endif  // IL_CUDA_DOT_H
