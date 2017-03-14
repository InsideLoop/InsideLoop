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

#include <il/core/base.h>
#include <il/String.h>

namespace il {

enum class ErrorCode {
  ok,
  unchecked,
  failed_precondition,
  already_there,
  not_found,
  file_not_found,
  wrong_rank,
  cannot_close_file,
  integer_overflow,
  bad_allocation,
  division_by_zero,
  negative_number,
  nonpositive_number,
  positive_number,
  nonnegative_number,
  no_convergence,
  wrong_file_format,
  file_format_version_not_supported,
  wrong_type,
  wrong_order,
  cannot_write_to_file,
  wrong_input,
  internal_error,
  unimplemented
};


class Status {
 private:
  il::ErrorCode error_code_;
  il::String message_;
  bool status_has_been_checked_;

 public:
  Status();
  ~Status();
  void set(ErrorCode code, const char* message);
  void set(ErrorCode error_code, const il::String& message);
  void set_error(ErrorCode code);
  void set_message(const char* message);
  void set_ok();
  ErrorCode error_code() const;
  const il::String& message() const;
  bool ok();
  void ignore_error();
  void abort_on_error();
};

inline Status::Status() : message_{} {
  status_has_been_checked_ = true;
  error_code_ = il::ErrorCode::ok;
}

inline Status::~Status() {
  if (!status_has_been_checked_) {
    il::abort();
  }
}

inline void Status::set_ok() {
  if (!status_has_been_checked_) {
    error_code_ = il::ErrorCode::unchecked;
  } else {
    status_has_been_checked_ = false;
    error_code_ = il::ErrorCode::ok;
  }
}

inline void Status::set(ErrorCode error_code, const char* message) {
  if (!status_has_been_checked_) {
    error_code_ = il::ErrorCode::unchecked;
  } else {
    status_has_been_checked_ = false;
    error_code_ = error_code;
  }
  message_ = il::String{message};
}

inline void Status::set(ErrorCode error_code, const il::String& message) {
  if (!status_has_been_checked_) {
    error_code_ = il::ErrorCode::unchecked;
  } else {
    status_has_been_checked_ = false;
    error_code_ = error_code;
  }
  message_ = message;
}

inline void Status::set_error(ErrorCode error_code) {
  if (!status_has_been_checked_) {
    error_code_ = il::ErrorCode::unchecked;
  } else {
    status_has_been_checked_ = false;
    error_code_ = error_code;
  }
}

inline void Status::set_message(const char* message) {
  message_ = il::String{message};
}

inline ErrorCode Status::error_code() const {
  return error_code_;
}

inline const il::String& Status::message() const {
  return message_;
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
    il::abort();
  }
}

}

#endif  // IL_ERROR_H
