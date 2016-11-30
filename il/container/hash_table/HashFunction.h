//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_HASHFUNCTION_H
#define IL_HASHFUNCTION_H

#include <il/base.h>

// include/ADT/DenseMapInfo.h

namespace il {

template <typename T>
class HashFunction {
 public:
  static inline T empty_key();
  static inline T tombstone_key();
  static unsigned hash_value(const T& val);
  static bool is_equal(const T& val0, const T& val1);
};

template <>
class HashFunction<char> {
 public:
  static inline char empty_key() { return ~0; }
  static inline char tombstone_key() { return ~0 - 1; }
  static unsigned hash_value(const char& val) {
    return static_cast<unsigned>(val * 37U);
  }
  static bool is_equal(const int& val0, const int& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<int> {
 public:
  static inline int empty_key() { return std::numeric_limits<int>::max(); }
  static inline int tombstone_key() { return std::numeric_limits<int>::min(); }
  static unsigned hash_value(const int& val) {
    return static_cast<unsigned>(val * 37U);
  }
  static bool is_equal(const int& val0, const int& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<long> {
 public:
  static inline long empty_key() { return std::numeric_limits<long>::max(); }
  static inline long tombstone_key() {
    return std::numeric_limits<long>::min();
  }
  static unsigned hash_value(const long& val) {
    return static_cast<unsigned>(val * 37UL);
  }
  static bool is_equal(const long& val0, const long& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<long long> {
 public:
  static inline long long empty_key() {
    return std::numeric_limits<long long>::max();
  }
  static inline long long tombstone_key() {
    return std::numeric_limits<long long>::min();
  }
  static unsigned hash_value(const long long& val) {
    return static_cast<unsigned>(val * 37ULL);
  }
  static bool is_equal(const long long& val0, const long long& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<unsigned> {
 public:
  static inline unsigned empty_key() {
    return std::numeric_limits<unsigned>::max();
  }
  static inline unsigned tombstone_key() {
    return std::numeric_limits<unsigned>::max() - 1;
  }
  static unsigned hash_value(const unsigned& val) {
    return static_cast<unsigned>(val * 37U);
  }
  static bool is_equal(const unsigned& val0, const unsigned& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<unsigned long> {
 public:
  static inline unsigned long empty_key() {
    return std::numeric_limits<unsigned long>::max();
  }
  static inline unsigned long tombstone_key() {
    return std::numeric_limits<unsigned long>::max() - 1;
  }
  static unsigned hash_value(const unsigned long& val) {
    return static_cast<unsigned>(val * 37UL);
  }
  static bool is_equal(const unsigned long& val0, const unsigned long& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<unsigned long long> {
 public:
  static inline unsigned long long empty_key() {
    return std::numeric_limits<unsigned long long>::max();
  }
  static inline unsigned long long tombstone_key() {
    return std::numeric_limits<unsigned long long>::max() - 1;
  }
  static unsigned hash_value(const unsigned long long& val) {
    return static_cast<unsigned>(val * 37ULL);
  }
  static bool is_equal(const unsigned long long& val0,
                       const unsigned long long& val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<std::string> {
 public:
  static inline std::string empty_key() {
    return std::string{"<<<EMPTY KEY>>>"};
  }
  static inline std::string tombstone_key() {
    return std::string{"<<TOMBSTONE KEY>>>"};
  }
  static unsigned hash_value(const std::string& val) {
    unsigned hash = 5381;
    for (std::size_t k = 0; k < val.size(); ++k) {
      hash = ((hash << 5) + hash) + val[k];
    }
    return hash;
  }
  static bool is_equal(const std::string& val0, const std::string& val1) {
    return val0 == val1;
  }
};
}

#endif  // IL_HASHFUNCTION_H
