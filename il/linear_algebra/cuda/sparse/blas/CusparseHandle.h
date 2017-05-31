//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CUSPARSEHANDLE_H
#define IL_CUSPARSEHANDLE_H

#include <cusparse_v2.h>

namespace il {

class CusparseHandle {
 private:
  cusparseHandle_t handle_;

 public:
  CusparseHandle();
  ~CusparseHandle();
  cusparseHandle_t handle();
};

inline CusparseHandle::CusparseHandle() : handle_{} {
  cusparseCreate(&handle_);
}

inline CusparseHandle::~CusparseHandle() { cusparseDestroy(handle_); }

inline cusparseHandle_t CusparseHandle::handle() { return handle_; }

}  // namespace il

#endif  // IL_CUSPARSEHANDLE_H
