//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_STRING_H
#define IL_STRING_H

// <cstring> is needed for memcpy
#include <cstring>
// <cstdint> is needed for std::uint8_t
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

class String {
 private:
  struct LargeString {
    std::uint8_t* data;
    std::size_t size;
    std::size_t capacity;
  };
  union {
    std::uint8_t data_[sizeof(LargeString)];
    LargeString large_;
  };

 public:
  String();
  String(const char* data);
  String(const char* data, il::int_t n);
  String(const il::String& s);
  String(il::String&& s);
  String& operator=(const il::String& s);
  String& operator=(il::String&& s);
  ~String();
  il::int_t size() const;
  il::int_t length() const;
  il::int_t capacity() const;
  bool is_empty() const;
  bool is_small() const;
  void reserve(il::int_t r);
  void append(const String& s);
  void append(const char* data);
  void append(const char*, il::int_t n);
  void append(char c);
  void append(std::int32_t cp);
  void append(il::int_t n, char c);
  void append(il::int_t n, std::int32_t cp);
  il::int_t next_cp(il::int_t i) const;
  std::int32_t to_cp(il::int_t i) const;
  const std::uint8_t* begin() const;
  const std::uint8_t* end() const;
  const char* c_string() const;
  bool operator==(const il::String& other) const;

 private:
  void set_small_size(il::int_t n);
  void set_large_capacity(il::int_t r);
  std::uint8_t* begin();
  std::uint8_t* end();
  bool valid_code_point(std::int32_t cp);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeString) - 1);
};

inline String::String() {
  data_[0] = static_cast<std::uint8_t>('\0');
  set_small_size(0);
}

inline String::String(const char* data) {
  IL_EXPECT_AXIOM("data is a UTF-8 null terminated string");

  il::int_t size = 0;
  while (data[size] != '\0') {
    ++size;
  }
  if (size <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = il::allocate_array<std::uint8_t>(size + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline String::String(const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(n) + 1);
    set_small_size(n);
  } else {
    large_.data = il::allocate_array<std::uint8_t>(n + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(n) + 1);
    large_.size = static_cast<std::size_t>(n);
    set_large_capacity(n);
  }
}

inline String::String(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = il::allocate_array<std::uint8_t>(size + 1);
    std::memcpy(large_.data, s.c_string(), static_cast<std::size_t>(size) + 1);
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline String::String(String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
    s.data_[0] = static_cast<std::uint8_t>('\0');
    s.set_small_size(0);
  }
}

inline String& String::operator=(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    if (!is_small()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.c_string(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!is_small()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocate_array<std::uint8_t>(size + 1);
      std::memcpy(large_.data, s.c_string(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
      set_large_capacity(size);
    }
  }
  return *this;
}

inline String& String::operator=(String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    if (size <= max_small_size_) {
      if (!is_small()) {
        il::deallocate(large_.data);
      }
      std::memcpy(data_, s.c_string(), static_cast<std::size_t>(size) + 1);
      set_small_size(size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.data_[0] = static_cast<std::uint8_t>('\0');
      s.set_small_size(0);
    }
  }
  return *this;
}

inline String::~String() {
  if (!is_small()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t String::size() const {
  if (is_small()) {
    return max_small_size_ - static_cast<il::int_t>(data_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

inline il::int_t String::length() const {
  il::int_t k = 0;
  for (il::int_t i = 0; i < size(); i = next_cp(i)) {
    ++k;
  }
  return k;
}

inline il::int_t String::capacity() const {
  if (is_small()) {
    return max_small_size_;
  } else {
    const std::uint8_t category_extract_mask = 0xC0;
    const std::size_t capacity_extract_mask =
        ~(static_cast<std::size_t>(category_extract_mask)
          << ((sizeof(std::size_t) - 1) * 8));
    return static_cast<il::int_t>(large_.capacity & capacity_extract_mask);
  }
}

inline void String::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const bool old_is_small = is_small();
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  std::uint8_t* new_data = il::allocate_array<std::uint8_t>(r + 1);
  std::memcpy(new_data, c_string(), static_cast<std::size_t>(old_size) + 1);
  if (!old_is_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  large_.size = old_size;
  set_large_capacity(r);
}

inline void String::append(const String& s) { append(s.c_string(), s.size()); }

inline void String::append(const char* data) {
  il::int_t size = 0;
  while (data[size] != '\0') {
    ++size;
  }
  append(data, size);
}

inline void String::append(char c) {
  IL_EXPECT_MEDIUM(static_cast<std::uint8_t>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  std::uint8_t* data = begin() + old_size;
  data[0] = static_cast<std::uint8_t>(c);
  data[1] = static_cast<std::uint8_t>('\0');
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(il::int_t n, char c) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(static_cast<std::uint8_t>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  std::uint8_t* data = begin() + old_size;
  for (il::int_t i = 0; i < n; ++i) {
    data[i] = static_cast<std::uint8_t>(c);
  }
  data[n] = static_cast<std::uint8_t>('\0');
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(std::int32_t cp) {
  IL_EXPECT_MEDIUM(valid_code_point(cp));

  const std::uint32_t ucp = static_cast<std::uint32_t>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00000080u) {
    new_size = old_size + 1;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    std::uint8_t* data = end();
    data[0] = static_cast<std::uint8_t>(ucp);
    data[1] = static_cast<std::uint8_t>('\0');
  } else if (ucp < 0x00000800u) {
    new_size = old_size + 2;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    std::uint8_t* data = end();
    data[0] = static_cast<std::uint8_t>((ucp >> 6) | 0x000000C0u);
    data[1] = static_cast<std::uint8_t>((ucp & 0x0000003Fu) | 0x00000080u);
    data[2] = static_cast<std::uint8_t>('\0');
  } else if (ucp < 0x00010000u) {
    new_size = old_size + 3;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    std::uint8_t* data = end();
    data[0] = static_cast<std::uint8_t>((ucp >> 12) | 0x000000E0u);
    data[1] =
        static_cast<std::uint8_t>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    data[2] = static_cast<std::uint8_t>((ucp & 0x0000003Fu) | 0x00000080u);
    data[3] = static_cast<std::uint8_t>('\0');
  } else {
    new_size = old_size + 4;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    std::uint8_t* data = end();
    data[0] = static_cast<std::uint8_t>((ucp >> 18) | 0x000000F0u);
    data[1] =
        static_cast<std::uint8_t>(((ucp >> 12) & 0x0000003Fu) | 0x00000080u);
    data[2] =
        static_cast<std::uint8_t>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    data[3] = static_cast<std::uint8_t>((ucp & 0x0000003Fu) | 0x00000080u);
    data[4] = static_cast<std::uint8_t>('\0');
  }
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(il::int_t n, std::int32_t cp) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(valid_code_point(cp));

  const std::uint32_t ucp = static_cast<std::uint32_t>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00000080u) {
    new_size = old_size + n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    std::uint8_t* data = end();
    const std::uint8_t cu0 = static_cast<std::uint8_t>(ucp);
    for (il::int_t i = 0; i < n; ++i) {
      data[i] = cu0;
    }
    data[n] = static_cast<std::uint8_t>('\0');
  } else if (ucp < 0x00000800u) {
    new_size = old_size + 2 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    std::uint8_t* data = end();
    const std::uint8_t cu0 =
        static_cast<std::uint8_t>((ucp >> 6) | 0x000000C0u);
    const std::uint8_t cu1 =
        static_cast<std::uint8_t>((ucp & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[2 * i] = cu0;
      data[2 * i + 1] = cu1;
    }
    data[2 * n] = static_cast<std::uint8_t>('\0');
  } else if (ucp < 0x00010000u) {
    new_size = old_size + 3 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    std::uint8_t* data = end();
    const std::uint8_t cu0 =
        static_cast<std::uint8_t>((ucp >> 12) | 0x000000E0u);
    const std::uint8_t cu1 =
        static_cast<std::uint8_t>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    const std::uint8_t cu2 =
        static_cast<std::uint8_t>((ucp & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[3 * i] = cu0;
      data[3 * i + 1] = cu1;
      data[3 * i + 2] = cu2;
    }
    data[3 * n] = static_cast<std::uint8_t>('\0');
  } else {
    new_size = old_size + 4 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    std::uint8_t* data = end();
    const std::uint8_t cu0 =
        static_cast<std::uint8_t>((ucp >> 18) | 0x000000F0u);
    const std::uint8_t cu1 =
        static_cast<std::uint8_t>(((ucp >> 12) & 0x0000003Fu) | 0x00000080u);
    const std::uint8_t cu2 =
        static_cast<std::uint8_t>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    const std::uint8_t cu3 =
        static_cast<std::uint8_t>((ucp & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[4 * i] = cu0;
      data[4 * i + 1] = cu1;
      data[4 * i + 2] = cu2;
      data[4 * i + 3] = cu3;
    }
    data[4 * n] = static_cast<std::uint8_t>('\0');
  }
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline const char* String::c_string() const {
  if (is_small()) {
    return reinterpret_cast<const char*>(data_);
  } else {
    return reinterpret_cast<const char*>(large_.data);
  }
}

inline bool String::is_empty() const { return size() == 0; }

inline il::int_t String::next_cp(il::int_t i) const {
  const std::uint8_t* data = begin();
  do {
    ++i;
  } while (i < size() && ((data[i] & 0xC0u) == 0x80u));
  return i;
}

inline std::int32_t String::to_cp(il::int_t i) const {
  std::uint32_t ans = 0;
  const std::uint8_t* data = begin();
  if ((data[i] & 0x80u) == 0) {
    ans = static_cast<std::uint32_t>(data[i]);
  } else if ((data[i] & 0xE0u) == 0xC0u) {
    ans = (static_cast<std::uint32_t>(data[i] & 0x1Fu) << 6) +
          static_cast<std::uint32_t>(data[i + 1] & 0x3Fu);
  } else if ((data[i] & 0xF0u) == 0xE0u) {
    ans = (static_cast<std::uint32_t>(data[i] & 0x0Fu) << 12) +
          (static_cast<std::uint32_t>(data[i + 1] & 0x3Fu) << 6) +
          static_cast<std::uint32_t>(data[i + 2] & 0x3Fu);
  } else {
    ans = (static_cast<std::uint32_t>(data[i] & 0x07u) << 18) +
          (static_cast<std::uint32_t>(data[i + 1] & 0x3Fu) << 12) +
          (static_cast<std::uint32_t>(data[i + 2] & 0x3Fu) << 6) +
          static_cast<std::uint32_t>(data[i + 3] & 0x3Fu);
  }
  return static_cast<std::int32_t>(ans);
}

inline bool String::is_small() const {
  const std::uint8_t category_extract_mask = 0xC0;
  return (data_[max_small_size_] & category_extract_mask) == 0;
}

inline bool String::operator==(const il::String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const std::uint8_t* p0 = begin();
    const std::uint8_t* p1 = other.begin();
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline void String::set_small_size(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  data_[max_small_size_] = static_cast<std::uint8_t>(max_small_size_ - n);
}

inline void String::set_large_capacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline const std::uint8_t* String::begin() const {
  if (is_small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline std::uint8_t* String::begin() {
  if (is_small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline const std::uint8_t* String::end() const {
  if (is_small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline std::uint8_t* String::end() {
  if (is_small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline void String::append(const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::int_t old_size = size();
  const il::int_t new_capacity =
      il::max(old_size + n, il::min(max_small_size_, 2 * old_size));
  reserve(new_capacity);

  if (is_small()) {
    std::memcpy(data_ + old_size, data, static_cast<std::size_t>(n));
    data_[old_size + n] = static_cast<std::uint8_t>('\0');
    set_small_size(old_size + n);
  } else {
    std::memcpy(large_.data + old_size, data, static_cast<std::size_t>(n));
    large_.data[old_size + n] = static_cast<std::uint8_t>('\0');
    large_.size = old_size + n;
  }
}

inline bool String::valid_code_point(std::int32_t cp) {
  const std::uint32_t code_point_max = 0x0010FFFFu;
  const std::uint32_t lead_surrogate_min = 0x0000D800u;
  const std::uint32_t lead_surrogate_max = 0x0000DBFFu;
  const std::uint32_t ucp = static_cast<std::uint32_t>(cp);
  return ucp <= code_point_max &&
         (ucp < lead_surrogate_min || ucp > lead_surrogate_max);
}

}  // namespace il

#endif  // IL_STRING_H
