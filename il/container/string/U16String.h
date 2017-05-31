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
  U16String(const char16_t* data);
  U16String(const il::U16String& s);
  U16String(il::U16String&& s);
  U16String& operator=(const il::U16String& s);
  U16String& operator=(il::U16String&& s);
  ~U16String();
  il::int_t size() const;
  il::int_t capacity() const;
  bool small() const;
  void reserve(il::int_t r);
  void append(const U16String& s);
  void append(const char16_t* data);
  void append(char c);
  void append(std::int32_t cp);
  void append(il::int_t n, char c);
  void append(il::int_t n, std::int32_t cp);
  il::int_t next_cp(il::int_t i) const;
  std::int32_t cp(il::int_t i) const;
  bool empty() const;
  const std::uint16_t* begin() const;
  const std::uint16_t* end() const;
  const char16_t* w_string() const;
  bool operator==(const il::U16String& other) const;

 private:
  void set_small_size(il::int_t n);
  void set_large_capacity(il::int_t r);
  std::uint16_t* begin();
  std::uint16_t* end();
  void append(const char16_t*, il::int_t n);
  bool valid_code_point(std::int32_t cp);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeU16String) / 2 - 1);
};

inline U16String::U16String() {
  data_[0] = static_cast<std::uint16_t>('\0');
  set_small_size(0);
}

inline U16String::U16String(const char16_t* data) {
  IL_EXPECT_AXIOM("data is a UTF-16 null terminated string");

  il::int_t size = 0;
  while (data[size] != static_cast<char16_t>('\0')) {
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
    std::memcpy(data_, s.w_string(), 2 * (static_cast<std::size_t>(size) + 1));
    set_small_size(size);
  } else {
    large_.data = il::allocate_array<std::uint16_t>(size + 1);
    std::memcpy(large_.data, s.w_string(),
                2 * (static_cast<std::size_t>(size) + 1));
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline U16String::U16String(U16String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.w_string(), 2 * (static_cast<std::size_t>(size) + 1));
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
    if (!small()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.w_string(), 2 * (static_cast<std::size_t>(size) + 1));
    set_small_size(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.w_string(),
                  2 * (static_cast<std::size_t>(size) + 1));
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!small()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocate_array<std::uint16_t>(size + 1);
      std::memcpy(large_.data, s.w_string(),
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
      if (!small()) {
        il::deallocate(large_.data);
      }
      std::memcpy(data_, s.w_string(),
                  2 * (static_cast<std::size_t>(size) + 1));
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
  if (!small()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t U16String::size() const {
  if (small()) {
    return max_small_size_ - static_cast<il::int_t>(data_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

inline il::int_t U16String::capacity() const {
  if (small()) {
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

  const bool old_is_small = small();
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  std::uint16_t* new_data = il::allocate_array<std::uint16_t>(r + 1);
  std::memcpy(new_data, w_string(),
              2 * (static_cast<std::size_t>(old_size) + 1));
  if (!old_is_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  large_.size = old_size;
  set_large_capacity(r);
}

inline void U16String::append(const U16String& s) {
  append(s.w_string(), s.size());
}

inline void U16String::append(const char16_t* data) {
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
  if (small()) {
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
  if (small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void U16String::append(std::int32_t cp) {
  IL_EXPECT_MEDIUM(valid_code_point(cp));

  const std::uint32_t ucp = static_cast<std::uint32_t>(cp);
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
    const std::uint32_t a = ucp - 0x00010000u;
    data[0] = static_cast<std::uint16_t>(a >> 10) + 0xD800u;
    data[1] = static_cast<std::uint16_t>(a & 0x3FF) + 0xDC00u;
    data[2] = static_cast<std::uint16_t>('\0');
  }
  if (small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void U16String::append(il::int_t n, std::int32_t cp) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(valid_code_point(cp));
}

inline const char16_t* U16String::w_string() const {
  if (small()) {
    return reinterpret_cast<const char16_t*>(data_);
  } else {
    return reinterpret_cast<const char16_t*>(large_.data);
  }
}

inline bool U16String::empty() const { return size() == 0; }

inline il::int_t U16String::next_cp(il::int_t i) const {
  const std::uint16_t* data = begin();
  if (data[i] < 0xD800u || data[i] >= 0xDC00u) {
    return i + 1;
  } else {
    return i + 2;
  }
}

inline std::int32_t U16String::cp(il::int_t i) const {
  const std::uint16_t* data = begin();
  if (data[i] < 0xD800u || data[i] >= 0xDC00u) {
    return static_cast<std::int32_t>(data[i]);
  } else {
    std::uint32_t a = static_cast<std::uint32_t>(data[i]);
    std::uint32_t b = static_cast<std::uint32_t>(data[i + 1]);
    return static_cast<std::int32_t>(((a & 0x03FFu) << 10) + (b & 0x03FFu) +
                                     0x00010000u);
  }
}

inline bool U16String::small() const {
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

inline const std::uint16_t* U16String::begin() const {
  if (small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline std::uint16_t* U16String::begin() {
  if (small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline const std::uint16_t* U16String::end() const {
  if (small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline std::uint16_t* U16String::end() {
  if (small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline void U16String::append(const char16_t* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::int_t old_size = size();
  const il::int_t new_capacity = il::max(old_size + n, 2 * old_size);
  reserve(new_capacity);

  if (small()) {
    std::memcpy(data_ + old_size, data, 2 * static_cast<std::size_t>(n));
    data_[old_size + n] = static_cast<std::uint16_t>('\0');
    set_small_size(old_size + n);
  } else {
    std::memcpy(large_.data + old_size, data, 2 * static_cast<std::size_t>(n));
    large_.data[old_size + n] = static_cast<std::uint16_t>('\0');
    large_.size = old_size + n;
  }
}

inline bool U16String::valid_code_point(std::int32_t cp) {
  const std::uint32_t code_point_max = 0x0010FFFFu;
  const std::uint32_t lead_surrogate_min = 0x0000D800u;
  const std::uint32_t lead_surrogate_max = 0x0000DBFFu;
  const std::uint32_t ucp = static_cast<std::uint32_t>(cp);
  return ucp <= code_point_max &&
         (ucp < lead_surrogate_min || ucp > lead_surrogate_max);
}
}  // namespace il

#endif  // IL_U16STRING_H
