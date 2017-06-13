//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_U16STRING_H
#define IL_U16STRING_H

// <cstring> is needed for memcpy
#include <cstring>
// <cstdint> is needed for std::uint16_t
#include <cstdint>

#include <il/base.h>
#include <il/core/memory/allocate.h>

namespace il {

// LargeString is defined this way (64-bit system, little endian as on Linux
// and Windows)
//
// - For small string optimization: 24 chars which makes 24 bytes
// - For large strings: 1 pointer and 2 std::size_t which make 24 bytes
//
// The last 2 bits (the most significant ones as we are on a little endian
// system), are used to know if:
//
// 00: small size optimization
// 01: large size (the string is on the heap)
// 10: not a String object, used as empty key for hash tables
// 11: not a String object, used as tombstone key for hash tables

#if defined(_WIN32) || defined(WIN32)
#define IL_U16CHAR wchar_t
#else
#define IL_U16CHAR char16_t
#endif

class U16String {
 private:
  struct LargeU16String {
    std::uint16_t* data;
    std::size_t size;
    std::size_t capacity;
  };
  union {
    std::uint16_t data_[sizeof(LargeU16String) / 2];
    LargeU16String large_;
  };

 public:
  U16String();
  U16String(const IL_U16CHAR* data);
  U16String(const il::U16String& s);
  U16String(il::U16String&& s);
  U16String& operator=(const il::U16String& s);
  U16String& operator=(il::U16String&& s);
  ~U16String();
  il::int_t size() const;
  il::int_t capacity() const;
  bool is_small() const;
  void reserve(il::int_t r);
  void append(const U16String& s);
  void append(const IL_U16CHAR* data);
  void append(char c);
  void append(int cp);
  void append(il::int_t n, char c);
  void append(il::int_t n, int cp);
  il::int_t next_code_point(il::int_t i) const;
  int cp(il::int_t i) const;
  bool is_empty() const;
  const std::uint16_t* begin() const;
  const std::uint16_t* end() const;
  const IL_U16CHAR* c16_string() const;
  bool operator==(const il::U16String& other) const;

 private:
  void set_small_size(il::int_t n);
  void set_large_capacity(il::int_t r);
  std::uint16_t* begin();
  std::uint16_t* end();
  void append(const IL_U16CHAR*, il::int_t n);
  bool valid_code_point(int cp);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeU16String) / 2 - 1);
};

inline U16String::U16String() {
  data_[0] = static_cast<std::uint16_t>('\0');
  set_small_size(0);
}

inline U16String::U16String(const IL_U16CHAR* data) {
  IL_EXPECT_AXIOM("data is a UTF-16 null terminated string");

  il::int_t size = 0;
  while (data[size] != static_cast<IL_U16CHAR>('\0')) {
    ++size;
  }
  if (size <= max_small_size_) {
    std::memcpy(data_, data, 2 * (static_cast<std::size_t>(size) + 1));
    set_small_size(size);
  } else {
    large_.data = il::allocate_array<std::uint16_t>(size + 1);
    std::memcpy(large_.data, data, 2 * (static_cast<std::size_t>(size) + 1));
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline U16String::U16String(const U16String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.begin(), 2 * (static_cast<std::size_t>(size) + 1));
    set_small_size(size);
  } else {
    large_.data = il::allocate_array<std::uint16_t>(size + 1);
    std::memcpy(large_.data, s.begin(),
                2 * (static_cast<std::size_t>(size) + 1));
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline U16String::U16String(U16String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.begin(), 2 * (static_cast<std::size_t>(size) + 1));
    set_small_size(size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
    s.data_[0] = static_cast<std::uint16_t>('\0');
    s.set_small_size(0);
  }
}

inline U16String& U16String::operator=(const U16String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    if (!is_small()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.begin(), 2 * (static_cast<std::size_t>(size) + 1));
    set_small_size(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.begin(),
                  2 * (static_cast<std::size_t>(size) + 1));
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!is_small()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocate_array<std::uint16_t>(size + 1);
      std::memcpy(large_.data, s.begin(),
                  2 * (static_cast<std::size_t>(size) + 1));
      large_.size = static_cast<std::size_t>(size);
      set_large_capacity(size);
    }
  }
  return *this;
}

inline U16String& U16String::operator=(U16String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    if (size <= max_small_size_) {
      if (!is_small()) {
        il::deallocate(large_.data);
      }
      std::memcpy(data_, s.begin(), 2 * (static_cast<std::size_t>(size) + 1));
      set_small_size(size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.data_[0] = static_cast<std::uint16_t>('\0');
      s.set_small_size(0);
    }
  }
  return *this;
}

inline U16String::~U16String() {
  if (!is_small()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t U16String::size() const {
  if (is_small()) {
    return max_small_size_ - static_cast<il::int_t>(data_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

inline il::int_t U16String::capacity() const {
  if (is_small()) {
    return max_small_size_;
  } else {
    const unsigned char category_extract_mask = 0xC0;
    const std::size_t capacity_extract_mask =
        ~(static_cast<std::size_t>(category_extract_mask)
          << ((sizeof(std::size_t) - 1) * 8));
    return static_cast<il::int_t>(large_.capacity & capacity_extract_mask);
  }
}

inline void U16String::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const bool old_is_small = is_small();
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  std::uint16_t* new_data = il::allocate_array<std::uint16_t>(r + 1);
  std::memcpy(new_data, begin(), 2 * (static_cast<std::size_t>(old_size) + 1));
  if (!old_is_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  large_.size = old_size;
  set_large_capacity(r);
}

inline void U16String::append(const U16String& s) {
  append(reinterpret_cast<const IL_U16CHAR*>(s.begin()), s.size());
}

inline void U16String::append(const IL_U16CHAR* data) {
  il::int_t size = 0;
  while (data[size] != static_cast<std::uint16_t>('\0')) {
    ++size;
  }
  append(data, size);
}

inline void U16String::append(char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  std::uint16_t* data = begin() + old_size;
  data[0] = static_cast<std::uint16_t>(c);
  data[1] = static_cast<std::uint16_t>('\0');
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void U16String::append(il::int_t n, char c) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  std::uint16_t* data = begin() + old_size;
  for (il::int_t i = 0; i < n; ++i) {
    data[i] = static_cast<std::uint16_t>(c);
  }
  data[n] = static_cast<std::uint16_t>('\0');
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void U16String::append(int cp) {
  IL_EXPECT_MEDIUM(valid_code_point(cp));

  const unsigned int ucp = static_cast<unsigned int>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00010000u) {
    new_size = old_size + 1;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    std::uint16_t* data = end();
    data[0] = static_cast<std::uint16_t>(ucp);
    data[1] = static_cast<std::uint16_t>('\0');
  } else {
    new_size = old_size + 2;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    std::uint16_t* data = end();
    const unsigned int a = ucp - 0x00010000u;
    data[0] = static_cast<std::uint16_t>(a >> 10) + 0xD800u;
    data[1] = static_cast<std::uint16_t>(a & 0x3FF) + 0xDC00u;
    data[2] = static_cast<std::uint16_t>('\0');
  }
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void U16String::append(il::int_t n, int cp) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(valid_code_point(cp));
}

inline const std::uint16_t* U16String::begin() const {
  if (is_small()) {
    return reinterpret_cast<const std::uint16_t*>(data_);
  } else {
    return reinterpret_cast<const std::uint16_t*>(large_.data);
  }
}

inline const IL_U16CHAR* U16String::c16_string() const {
  if (is_small()) {
    return reinterpret_cast<const IL_U16CHAR*>(data_);
  } else {
    return reinterpret_cast<const IL_U16CHAR*>(large_.data);
  }
}

inline bool U16String::is_empty() const { return size() == 0; }

inline il::int_t U16String::next_code_point(il::int_t i) const {
  const std::uint16_t* data = begin();
  if (data[i] < 0xD800u || data[i] >= 0xDC00u) {
    return i + 1;
  } else {
    return i + 2;
  }
}

inline int U16String::cp(il::int_t i) const {
  const std::uint16_t* data = begin();
  if (data[i] < 0xD800u || data[i] >= 0xDC00u) {
    return static_cast<int>(data[i]);
  } else {
    unsigned int a = static_cast<unsigned int>(data[i]);
    unsigned int b = static_cast<unsigned int>(data[i + 1]);
    return static_cast<int>(((a & 0x03FFu) << 10) + (b & 0x03FFu) +
                            0x00010000u);
  }
}

inline bool U16String::operator==(const il::U16String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const std::uint16_t* p0 = begin();
    const std::uint16_t* p1 = other.begin();
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool U16String::is_small() const {
  const std::uint16_t category_extract_mask = 0xC000;
  return (data_[max_small_size_] & category_extract_mask) == 0;
}

inline void U16String::set_small_size(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  data_[max_small_size_] = static_cast<std::uint16_t>(max_small_size_ - n);
}

inline void U16String::set_large_capacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline std::uint16_t* U16String::begin() {
  if (is_small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline const std::uint16_t* U16String::end() const {
  if (is_small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline std::uint16_t* U16String::end() {
  if (is_small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline void U16String::append(const IL_U16CHAR* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::int_t old_size = size();
  const il::int_t new_capacity = il::max(old_size + n, 2 * old_size);
  reserve(new_capacity);

  if (is_small()) {
    std::memcpy(data_ + old_size, data, 2 * static_cast<std::size_t>(n));
    data_[old_size + n] = static_cast<std::uint16_t>('\0');
    set_small_size(old_size + n);
  } else {
    std::memcpy(large_.data + old_size, data, 2 * static_cast<std::size_t>(n));
    large_.data[old_size + n] = static_cast<std::uint16_t>('\0');
    large_.size = old_size + n;
  }
}

inline bool U16String::valid_code_point(int cp) {
  const unsigned int code_point_max = 0x0010FFFFu;
  const unsigned int lead_surrogate_min = 0x0000D800u;
  const unsigned int lead_surrogate_max = 0x0000DBFFu;
  const unsigned int ucp = static_cast<unsigned int>(cp);
  return ucp <= code_point_max &&
         (ucp < lead_surrogate_min || ucp > lead_surrogate_max);
}
}  // namespace il

#endif  // IL_U16STRING_H
