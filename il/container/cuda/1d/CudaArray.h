//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUDAARRAY_H
#define IL_CUDAARRAY_H

#include <cuda_runtime_api.h>
#include <cuda.h>

// <initializer_list> is needed for std::initializer_list<T>
#include <initializer_list>

#include <il/base.h>

namespace il {

template<typename T>
class CudaArray {
 private:
  T *data_;
  int size_;
  int capacity_;

 public:
  /* \brief Construct an array of n elements
  */
  explicit CudaArray(int n);

  /* \brief The destructor
  */
  ~CudaArray();

  /* \brief Get the size of the il::CudaArray<T>
  */
  int size() const;

  /* \brief Get a pointer to the first element of the array
  */
  const T *data() const;

  /* \brief Get a pointer to the first element of the array
  */
  T *data();
};

template<typename T>
CudaArray<T>::CudaArray(int n) {
  IL_ASSERT_PRECOND(n >= 0);

  cudaMalloc(&data_, n * sizeof(T));
  size_ = n;
  capacity_ = n;
}

template<typename T>
CudaArray<T>::~CudaArray() {
  cudaFree(data_);
}

template<typename T>
int CudaArray<T>::size() const {
  return size_;
}

template<typename T>
const T* CudaArray<T>::data() const {
  return data_;
}

template<typename T>
T* CudaArray<T>::data() {
  return data_;
}

}
#endif  // IL_CUDAARRAY_H

