//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUDAARRAY2D_H
#define IL_CUDAARRAY2D_H

#include <cuda_runtime_api.h>
#include <cuda.h>

// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/base.h>

namespace il {

template<typename T>
class CudaArray2D {
 private:
  T *data_;
  int size_[2];
  int capacity_[2];

 public:
  /* \brief Construct an array of n elements
  */
  explicit CudaArray2D(int n0, int n1);

  /* \brief The destructor
  */
  ~CudaArray2D();

  /* \brief Get the size of the il::CudaArray2D<T>
  */
  int size(int d) const;

  /* \brief Get a pointer to the first element of the array
  */
  const T *data() const;

  /* \brief Get a pointer to the first element of the array
  */
  T *data();
};

template<typename T>
CudaArray2D<T>::CudaArray2D(int n0, int n1) {
  IL_ASSERT_PRECOND(n0 >= 0);
  IL_ASSERT_PRECOND(n1 >= 0);

  cudaMalloc(&data_, n0 * n1 * sizeof(T));
  size_[0] = n0;
  size_[1] = n1;
  capacity_[0] = n0;
  capacity_[1] = n1;
}

template<typename T>
CudaArray2D<T>::~CudaArray2D() {
  cudaFree(data_);
}

template<typename T>
int CudaArray2D<T>::size(int d) const {
  return size_[d];
}

template<typename T>
const T* CudaArray2D<T>::data() const {
  return data_;
}

template<typename T>
T* CudaArray2D<T>::data() {
  return data_;
}

}
#endif  // IL_CUDAARRAY_H
