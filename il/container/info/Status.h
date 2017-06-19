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

enum class ErrorDomain : unsigned short {
  kFilesystem = 0,
  kBinary = 4,
  kParse = 1,
  kOverflow = 2,
  kFloatingPoint = 3,
  kMatrix = 5,
  kUnimplemented = 126,
  kUndefined = 127
};

enum class Error : unsigned short {
  kFilesystemFileNotFound = 0 * 256 + 0,
  kFilesystemDirectoryNotFound = 0 * 256 + 1,
  kFilesystemNoReadAccess = 0 * 256 + 2,
  kFilesystemNoWriteAccess = 0 * 256 + 3,
  kFilesystemCanNotCloseFile = 0 * 256 + 4,
  kFilesystemCanNotWriteToFile = 0 * 256 + 5,
  //
  kBinaryFileWrongFormat = 4 * 256 + 0,
  kBinaryFileWrongType = 4 * 256 + 1,
  kBinaryFileWrongRank = 4 * 256 + 2,
  kBinaryFileWrongEndianness = 4 * 256 + 3,
  //
  kParseBool = 1 * 256 + 0,
  kParseNumber = 1 * 256 + 11,
  kParseInt = 1 * 256 + 1,
  kParseIntOverflow = 1 * 256 + 2,
  kParseInteger = 1 * 256 + 3,
  kParseIntegerOverflow = 1 * 256 + 4,
  kParseFloat = 1 * 256 + 5,
  kParseDouble = 1 * 256 + 6,
  kParseString = 1 * 256 + 7,
  kParseUnclosedArray = 1 * 256 + 8,
  kParseUnidentifiedTrailingCharacter = 1 * 256 + 9,
  kParseCanNotDetermineType = 1 * 256 + 10,
  kParseHeterogeneousArray = 1 * 256 + 12,
  kParseArray = 1 * 256 + 13,
  kParseTable = 1 * 256 + 14,
  kParseDuplicateKey = 1 * 256 + 14,
  kParseKey = 1 * 256 + 15,
  kParseValue = 1 * 256 + 16,
  //
  kOverflowInt = 2 * 256 + 0,
  kOverflowInteger = 2 * 256 + 1,
  //
  kFloatingPointNonNegative = 3 * 256 + 0,
  kFloatingPointPositive = 3 * 256 + 1,
  kFloatingPointNonPositive = 3 * 256 + 2,
  kFloatingPointNegative = 3 * 256 + 3,
  //
  kMatrixSingular = 5 * 256 + 0,
  kMatrixEigenValueNoConvergence = 5 * 256 + 1,
  //
  kUnimplemented = 126 * 256 + 0,
  //
  kUndefined = 127 * 256 + 0
};

inline il::ErrorDomain domain(il::Error error) {
  return static_cast<il::ErrorDomain>(static_cast<short>(error) >> 8);
}

#define IL_SET_SOURCE(status) status.setSource(__FILE__, __LINE__)

class Status {
 private:
  bool ok_;
  bool to_check_;
  Error error_;
  il::Info info_;

 public:
  Status();
  Status(const Status& other) = delete;
  Status(Status&& other);
  Status& operator=(const Status& other) = delete;
  Status& operator=(Status&& other);
  ~Status();
  void setOk();
  void setError(il::Error error);
  void setInfo(const char* key, int value);
#ifdef IL_64_BIT
  void setInfo(const char* key, il::int_t value);
#endif
  void setInfo(const char* key, double value);
  void setInfo(const char* key, const char* value);
  void setSource(const char* file, il::int_t line);
  il::int_t toInteger(const char* key) const;
  double toDouble(const char* key) const;
  const char* asCString(const char* key) const;
  void rearm();
  bool ok();
  void abortOnError();
  void ignoreError();
  il::Error error() const;
  il::ErrorDomain domain() const;
};

inline Status::Status() : info_{} {
  ok_ = true;
  to_check_ = false;
  error_ = il::Error::kUndefined;
}

inline Status::Status(Status&& other) {
  ok_ = other.ok_;
  to_check_ = true;
  error_ = other.error_;
  info_ = other.info_;

  other.to_check_ = false;
}

inline Status& Status::operator=(Status&& other) {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = other.ok_;
  to_check_ = true;
  error_ = other.error_;
  info_ = other.info_;

  other.to_check_ = false;

  return *this;
}

inline Status::~Status() { IL_EXPECT_MEDIUM(!to_check_); }

inline void Status::setError(il::Error error) {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = false;
  to_check_ = true;
  error_ = error;
  info_.clear();
}

inline void Status::setInfo(const char* key, int value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.set(key, value);
}

#ifdef IL_64_BIT
inline void Status::setInfo(const char* key, il::int_t value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.set(key, value);
}
#endif

inline void Status::setInfo(const char* key, double value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.set(key, value);
}

inline void Status::setInfo(const char* key, const char* value) {
  IL_EXPECT_MEDIUM(to_check_);
  IL_EXPECT_MEDIUM(!ok_);

  info_.set(key, value);
}

inline void Status::setSource(const char* file, il::int_t line) {
  setInfo("source_file", file);
  setInfo("source_line", line);
}

inline il::int_t Status::toInteger(const char* key) const {
  IL_EXPECT_MEDIUM(!ok_);

  return info_.toInteger(key);
}

inline double Status::toDouble(const char* key) const {
  IL_EXPECT_MEDIUM(!ok_);

  return info_.toDouble(key);
}

inline const char* Status::asCString(const char* key) const {
  IL_EXPECT_MEDIUM(!ok_);

  return info_.asCString(key);
}

inline void Status::setOk() {
  IL_EXPECT_MEDIUM(!to_check_);

  ok_ = true;
  to_check_ = true;
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

inline void Status::abortOnError() {
  IL_EXPECT_MEDIUM(to_check_);

  to_check_ = false;
  if (!ok_) {
    il::abort();
  }
}

inline void Status::ignoreError() {
  IL_EXPECT_MEDIUM(to_check_);

  to_check_ = false;
}

inline il::Error Status::error() const {
  IL_EXPECT_MEDIUM(!to_check_);

  return error_;
}
}  // namespace il

#endif  // IL_STATUS_H
