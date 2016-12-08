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
  il::ErrorCode code_;
  bool error_has_been_checked_;

 public:
  Error();
  ~Error();
  void set(ErrorCode code);
  ErrorCode code() const;
  bool raised();
  void ignore();
  void abort();
};

inline Error::Error() {
  error_has_been_checked_ = true;
  code_ = il::ErrorCode::ok;
}

inline Error::~Error() {
  if (!error_has_been_checked_) {
    IL_ASSERT(false);
  }
}

inline void Error::set(ErrorCode code) {
  if (!error_has_been_checked_) {
    code_ = il::ErrorCode::unchecked;
  } else {
    error_has_been_checked_ = false;
    code_ = code;
  }
}

inline ErrorCode Error::code() const {
  return code_;
}

inline bool Error::raised() {
  error_has_been_checked_ = true;
  return code_ != il::ErrorCode::ok;
}

inline void Error::ignore() {
  error_has_been_checked_ = true;
}

inline void Error::abort() {
  error_has_been_checked_ = true;
  if (code_ != il::ErrorCode::ok) {
    IL_ASSERT(false);
  }
}

}

#endif  // IL_ERROR_H
