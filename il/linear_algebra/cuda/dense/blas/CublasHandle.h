//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUBLAS_HANDLE_H
#define IL_CUBLAS_HANDLE_H

#include <cublas_v2.h>

namespace il {

class CublasHandle {
 private:
  cublasHandle_t handle_;

 public:
  CublasHandle();
  ~CublasHandle();
  cublasHandle_t handle();
};

inline CublasHandle::CublasHandle() : handle_{} { cublasCreate(&handle_); }

inline CublasHandle::~CublasHandle() { cublasDestroy(handle_); }

cublasHandle_t CublasHandle::handle() { return handle_; }

}  // namespace il

#endif  // IL_CUBLAS_HANDLE_H
