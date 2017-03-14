//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_STATUS_H
#define IL_STATUS_H

#include <il/Info.h>

namespace il {

enum class ErrorDomain : short {
  filesystem = 0,
  parse = 1
};

enum class Error : short {
  filesystem_file_not_found = 0 * 256 + 0,
  filesystem_directory_not_found,
  filesystem_no_read_access,
  filesystem_no_write_access,
  filesystem_cannot_close_file,
  parse_bool,
  parse_int,
  parse_int_overflow,
  parse_integer,
  parse_integer_overflow,
  parse_float,
  parse_double,
  parse_string,
  overflow_int,
  overflow_integer,
  unimplemented,
  undefined
};

class Status {
 private:
  bool ok_;
  bool to_check_;
  Error error_;
 public:
  il::Info info;

 public:
  Status();
  Status(const Status& other) = delete;
  Status(Status&& other);
  Status& operator=(const Status& other) = delete;
  Status& operator=(Status&& other);
  ~Status();
  void set(il::Error error);
  void set_ok();
  void rearm();
  bool ok();
  il::Error error() const;
  il::ErrorDomain domain() const;
};

inline Status::Status() : info{} {
  ok_ = true;
  to_check_ = false;
  error_ = il::Error::undefined;
}

inline Status::Status(Status&& other) {
  ok_ = other.ok_;
  to_check_ = true;
  error_ = other.error_;

  other.to_check_ = false;
}

inline Status& Status::operator=(Status&& other) {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = other.ok_;
  to_check_ = true;
  error_ = other.error_;

  other.to_check_ = false;

  return *this;
}

inline Status::~Status() {
  IL_EXPECT_MEDIUM(!to_check_);
}

inline void Status::set(il::Error error) {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = false;
  to_check_ = true;
  error_ = error;
  info.clear();
}

inline void Status::set_ok() {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = true;
  to_check_ = true;
  error_ = il::Error::undefined;
  info.clear();
}

inline void Status::rearm() {
  IL_EXPECT_MEDIUM(!to_check_);

  to_check_ = true;
}

inline bool Status::ok() {
  IL_EXPECT_MEDIUM(to_check_);

  to_check_ = false;
  return ok_;
}

inline il::Error Status::error() const {
  IL_EXPECT_MEDIUM(!to_check_);

  return error_;
}

}

#endif  // IL_STATUS_H

