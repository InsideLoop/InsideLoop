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

// <cstdlib> is used for std::abort
#include <cstdlib>

namespace il {

enum class ErrorCode {
  ok,
  unchecked,
  failed_precondition,
  already_there,
  not_found,
  integer_overflow,
  bad_allocation,
  division_by_zero,
  negative_number,
  nonpositive_number,
  positive_number,
  nonnegative_number,
  no_convergence,
  wrong_file_format,
  wrong_type,
  wrong_input,
  internal_error,
  unimplemented
};

class Status {
 private:
  il::ErrorCode error_code_;
  bool status_has_been_checked_;

 public:
  Status();
  ~Status();
  void set(ErrorCode code);
  ErrorCode error_code() const;
  bool ok();
  void ignore_error();
  void abort_on_error();
};

inline Status::Status() {
  status_has_been_checked_ = true;
  error_code_ = il::ErrorCode::ok;
}

inline Status::~Status() {
  if (!status_has_been_checked_) {
    std::abort();
  }
}

inline void Status::set(ErrorCode error_code) {
  if (!status_has_been_checked_) {
    error_code_ = il::ErrorCode::unchecked;
  } else {
    status_has_been_checked_ = false;
    error_code_ = error_code;
  }
}

inline ErrorCode Status::error_code() const {
  return error_code_;
}

inline bool Status::ok() {
  status_has_been_checked_ = true;
  return error_code_ == il::ErrorCode::ok;
}

inline void Status::ignore_error() {
  status_has_been_checked_ = true;
}

inline void Status::abort_on_error() {
  status_has_been_checked_ = true;
  if (error_code_ != il::ErrorCode::ok) {
    std::abort();
  }
}

}

#endif  // IL_ERROR_H
