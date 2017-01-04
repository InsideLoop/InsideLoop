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

namespace il {

template <typename T>
class HashFunction {
 public:
  static inline T empty_key();
  static inline T tombstone_key();
  static std::size_t hash_value(const T& val);
  static bool is_equal(const T& val0, const T& val1);
};

template <>
class HashFunction<char> {
 public:
  static inline char empty_key() { return ~0; }
  static inline char tombstone_key() { return ~0 - 1; }
  static std::size_t hash_value(char val) {
    return static_cast<std::size_t>(val * 37U);
  }
  static bool is_equal(char val0, char val1) {
    return val0 == val1;
  }
};

template <>
class HashFunction<int> {
 public:
  static inline int empty_key() { return std::numeric_limits<int>::max(); }
  static inline int tombstone_key() { return std::numeric_limits<int>::min(); }
  static std::size_t hash_value(int val) {
    return static_cast<std::size_t>(val * 37U);
  }
  static bool is_equal(int val0, int val1) {
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
  static std::size_t hash_value(long val) {
    return static_cast<std::size_t>(val * 37);
  }
  static bool is_equal(long val0, long val1) {
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
  static std::size_t hash_value(long long val) {
    return static_cast<std::size_t>(val * 37ULL);
  }
  static bool is_equal(long long val0, long long val1) {
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
  static std::size_t hash_value(unsigned val) {
    return static_cast<std::size_t>(val * 37U);
  }
  static bool is_equal(unsigned val0, unsigned val1) {
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
  static std::size_t hash_value(unsigned long val) {
    return static_cast<std::size_t>(val * 37UL);
  }
  static bool is_equal(unsigned long val0, unsigned long val1) {
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
  static std::size_t hash_value(unsigned long long val) {
    return static_cast<std::size_t>(val * 37ULL);
  }
  static bool is_equal(unsigned long long val0,
                       unsigned long long val1) {
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
  static std::size_t hash_value(const std::string& val) {
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

inline std::size_t unaligned_load(const char* p) {
  std::size_t result;
  __builtin_memcpy(&result, p, sizeof(result));
  return result;
}

inline std::size_t load_bytes(const char* p, int n) {
  std::size_t result = 0;
  --n;
  do
    result = (result << 8) + static_cast<unsigned char>(p[n]);
  while (--n >= 0);
  return result;
}

inline std::size_t shift_mix(std::size_t v) { return v ^ (v >> 47); }

std::size_t hash_bytes(const void* ptr, std::size_t len, std::size_t seed) {
  static const std::size_t mul =
      (static_cast<std::size_t>(0xc6a4a793UL) << 32UL) + (size_t)0x5bd1e995UL;
  const char* const buf = static_cast<const char*>(ptr);

  // Remove the bytes not divisible by the sizeof(size_t).  This
  // allows the main loop to process the data as 64-bit integers.
  const int len_aligned = len & ~0x7;
  const char* const end = buf + len_aligned;
  size_t hash = seed ^ (len * mul);
  for (const char* p = buf; p != end; p += 8) {
    const size_t data = shift_mix(unaligned_load(p) * mul) * mul;
    hash ^= data;
    hash *= mul;
  }
  if ((len & 0x7) != 0) {
    const size_t data = load_bytes(end, len & 0x7);
    hash ^= data;
    hash *= mul;
  }
  hash = shift_mix(hash) * mul;
  hash = shift_mix(hash);
  return hash;
}

template <>
class HashFunction<double> {
 public:
  static inline double empty_key() {
    return std::numeric_limits<double>::denorm_min();
  }
  static inline double tombstone_key() {
    return -std::numeric_limits<double>::denorm_min();
  }
  static std::size_t hash_value(const double& val) {
    (void)val;
    return (val != 0.0) ? hash_bytes(&val, sizeof(double),
                                     static_cast<size_t>(0xc70f6907UL))
                        : 0;
  }
  static bool is_equal(const double& val0,
                       const double& val1) {
    return val0 == val1;
  }
};
}

#endif  // IL_HASHFUNCTION_H
