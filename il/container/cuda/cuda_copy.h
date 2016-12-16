//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUDA_COPY_H
#define IL_CUDA_COPY_H

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/container/cuda/1d/CudaArray.h>
#include <il/container/cuda/2d/CudaArray2D.h>

namespace il {

template <typename T>
void copy(const il::Array<T>& src, il::io_t, il::CudaArray<T>& dest) {
  IL_ASSERT_PRECOND(src.size() == dest.size());

  cudaError_t error = cudaMemcpy(
      dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
  IL_ASSERT(error == 0);
}

template <typename T>
void copy(const il::CudaArray<T>& src, il::io_t, il::Array<T>& dest) {
  IL_ASSERT_PRECOND(src.size() == dest.size());

  cudaError_t error = cudaMemcpy(
      dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);
  IL_ASSERT(error == 0);
}

template <typename T>
void copy(const il::Array2D<T>& src, il::io_t, il::CudaArray2D<T>& dest) {
  IL_ASSERT_PRECOND(src.size(0) == dest.size(0));
  IL_ASSERT_PRECOND(src.size(1) == dest.size(1));

  cudaError_t error =
      cudaMemcpy(dest.data(), src.data(), src.size(0) * src.size(1) * sizeof(T),
                 cudaMemcpyHostToDevice);
  IL_ASSERT(error == 0);
}

template <typename T>
void copy(const il::CudaArray2D<T>& src, il::io_t, il::Array2D<T>& dest) {
  IL_ASSERT_PRECOND(src.size(0) == dest.size(0));
  IL_ASSERT_PRECOND(src.size(1) == dest.size(1));

  cudaError_t error =
      cudaMemcpy(dest.data(), src.data(), src.size(0) * src.size(1) * sizeof(T),
                 cudaMemcpyDeviceToHost);
  IL_ASSERT(error == 0);
}

}

#endif  // IL_CUDA_COPY_H
