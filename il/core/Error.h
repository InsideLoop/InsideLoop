//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ERROR_H
#define IL_ERROR_H

#include <string>

#include <il/core/ilassert.h>
#include <il/core/ildef.h>

namespace il {

enum class ErrorCode {
  ok,
  unchecked,
  failed_precondition,
  already_there,
  not_found,
  division_by_zero,
  negative_number,
  wrong_file_format,
  wrong_type,
  internal_error,
  unimplemented
};

class Error {
 private:
  bool error_has_been_checked_;
  il::ErrorCode code_;

 public:
  Error() {
    error_has_been_checked_ = true;
    code_ = il::ErrorCode::ok;
  }
  ~Error() {
    if (!error_has_been_checked_) {
      IL_ASSERT(false);
    }
  }
  void set(ErrorCode code) {
    if (!error_has_been_checked_) {
      code_ = il::ErrorCode::unchecked;
    } else {
      error_has_been_checked_ = false;
      code_ = code;
    }
  }
  ErrorCode code() {
    return code_;
  }
  bool raised() {
    error_has_been_checked_ = true;
    return code_ != il::ErrorCode::ok;
  }
  void ignore() {
    error_has_been_checked_ = true;
  }
  void abort() {
    error_has_been_checked_ = true;
    if (code_ != il::ErrorCode::ok) {
      IL_ASSERT(false);
    }
  }
};
}

#endif  // IL_ERROR_H
