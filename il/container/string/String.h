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

#include <il/base.h>
#include <il/core/math/safe_arithmetic.h>
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
    unsigned char* data;
    std::size_t size;
    std::size_t capacity;
  };
  union {
    unsigned char data_[sizeof(LargeString)];
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
  il::int_t capacity() const;
  bool is_empty() const;
  bool is_small() const;
  void reserve(il::int_t r);
  void append(const String& s);
  void append(const char* data);
  void append(const char* data, il::int_t n);
  void append(char c);
  void append(int cp);
  void append(il::int_t n, char c);
  void append(il::int_t n, int cp);
  il::int_t next_code_point(il::int_t i) const;
  // il::int_t next_line(il::int_t i) const;
  int to_code_point(il::int_t i) const;
  const unsigned char* begin() const;
  const unsigned char* end() const;
  const char* as_c_string() const;
  const unsigned char* data() const;
  bool operator==(const il::String& other) const;
  bool has_suffix(const char* data) const;

 private:
  void set_small_size(il::int_t n);
  void set_large_capacity(il::int_t r);
  unsigned char* begin();
  unsigned char* end();
  bool valid_code_point(int cp);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeString) - 1);
};

inline String::String() {
  data_[0] = static_cast<unsigned char>('\0');
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
    large_.data = il::allocate_array<unsigned char>(size + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline String::String(const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(n));
    data_[n] = '\0';
    set_small_size(n);
  } else {
    large_.data = il::allocate_array<unsigned char>(n + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(n));
    large_.data[n] = '\0';
    large_.size = static_cast<std::size_t>(n);
    set_large_capacity(n);
  }
}

inline String::String(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.as_c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = il::allocate_array<unsigned char>(size + 1);
    std::memcpy(large_.data, s.as_c_string(),
                static_cast<std::size_t>(size) + 1);
    large_.size = static_cast<std::size_t>(size);
    set_large_capacity(size);
  }
}

inline String::String(String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.as_c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
    s.data_[0] = static_cast<unsigned char>('\0');
    s.set_small_size(0);
  }
}

inline String& String::operator=(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    if (!is_small()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.as_c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.as_c_string(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!is_small()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocate_array<unsigned char>(size + 1);
      std::memcpy(large_.data, s.as_c_string(),
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
      std::memcpy(data_, s.as_c_string(), static_cast<std::size_t>(size) + 1);
      set_small_size(size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.data_[0] = static_cast<unsigned char>('\0');
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

inline il::int_t String::capacity() const {
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

inline void String::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const bool old_is_small = is_small();
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  unsigned char* new_data = il::allocate_array<unsigned char>(r + 1);
  std::memcpy(new_data, as_c_string(), static_cast<std::size_t>(old_size) + 1);
  if (!old_is_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  large_.size = old_size;
  set_large_capacity(r);
}

inline void String::append(const String& s) {
  append(s.as_c_string(), s.size());
}

inline void String::append(const char* data) {
  il::int_t size = 0;
  while (data[size] != '\0') {
    ++size;
  }
  append(data, size);
}

inline void String::append(char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  unsigned char* data = begin() + old_size;
  data[0] = static_cast<unsigned char>(c);
  data[1] = static_cast<unsigned char>('\0');
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(il::int_t n, char c) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  unsigned char* data = begin() + old_size;
  for (il::int_t i = 0; i < n; ++i) {
    data[i] = static_cast<unsigned char>(c);
  }
  data[n] = static_cast<unsigned char>('\0');
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(int cp) {
  IL_EXPECT_MEDIUM(valid_code_point(cp));

  const unsigned int ucp = static_cast<unsigned int>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00000080u) {
    new_size = old_size + 1;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = end();
    data[0] = static_cast<unsigned char>(ucp);
    data[1] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00000800u) {
    new_size = old_size + 2;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = end();
    data[0] = static_cast<unsigned char>((ucp >> 6) | 0x000000C0u);
    data[1] = static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    data[2] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00010000u) {
    new_size = old_size + 3;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = end();
    data[0] = static_cast<unsigned char>((ucp >> 12) | 0x000000E0u);
    data[1] =
        static_cast<unsigned char>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    data[2] = static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    data[3] = static_cast<unsigned char>('\0');
  } else {
    new_size = old_size + 4;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = end();
    data[0] = static_cast<unsigned char>((ucp >> 18) | 0x000000F0u);
    data[1] =
        static_cast<unsigned char>(((ucp >> 12) & 0x0000003Fu) | 0x00000080u);
    data[2] =
        static_cast<unsigned char>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    data[3] = static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    data[4] = static_cast<unsigned char>('\0');
  }
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(il::int_t n, int cp) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(valid_code_point(cp));

  const unsigned int ucp = static_cast<unsigned int>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00000080u) {
    new_size = old_size + n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = end();
    const unsigned char cu0 = static_cast<unsigned char>(ucp);
    for (il::int_t i = 0; i < n; ++i) {
      data[i] = cu0;
    }
    data[n] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00000800u) {
    new_size = old_size + 2 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = end();
    const unsigned char cu0 =
        static_cast<unsigned char>((ucp >> 6) | 0x000000C0u);
    const unsigned char cu1 =
        static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[2 * i] = cu0;
      data[2 * i + 1] = cu1;
    }
    data[2 * n] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00010000u) {
    new_size = old_size + 3 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = end();
    const unsigned char cu0 =
        static_cast<unsigned char>((ucp >> 12) | 0x000000E0u);
    const unsigned char cu1 =
        static_cast<unsigned char>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    const unsigned char cu2 =
        static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[3 * i] = cu0;
      data[3 * i + 1] = cu1;
      data[3 * i + 2] = cu2;
    }
    data[3 * n] = static_cast<unsigned char>('\0');
  } else {
    new_size = old_size + 4 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = end();
    const unsigned char cu0 =
        static_cast<unsigned char>((ucp >> 18) | 0x000000F0u);
    const unsigned char cu1 =
        static_cast<unsigned char>(((ucp >> 12) & 0x0000003Fu) | 0x00000080u);
    const unsigned char cu2 =
        static_cast<unsigned char>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    const unsigned char cu3 =
        static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[4 * i] = cu0;
      data[4 * i + 1] = cu1;
      data[4 * i + 2] = cu2;
      data[4 * i + 3] = cu3;
    }
    data[4 * n] = static_cast<unsigned char>('\0');
  }
  if (is_small()) {
    set_small_size(new_size);
  } else {
    large_.size = new_size;
  }
}

inline const char* String::as_c_string() const {
  if (is_small()) {
    return reinterpret_cast<const char*>(data_);
  } else {
    return reinterpret_cast<const char*>(large_.data);
  }
}

inline bool String::is_empty() const { return size() == 0; }

inline il::int_t String::next_code_point(il::int_t i) const {
  const unsigned char* data = begin();
  do {
    ++i;
  } while (i < size() && ((data[i] & 0xC0u) == 0x80u));
  return i;
}

inline int String::to_code_point(il::int_t i) const {
  unsigned int ans = 0;
  const unsigned char* data = begin();
  if ((data[i] & 0x80u) == 0) {
    ans = static_cast<unsigned int>(data[i]);
  } else if ((data[i] & 0xE0u) == 0xC0u) {
    ans = (static_cast<unsigned int>(data[i] & 0x1Fu) << 6) +
          static_cast<unsigned int>(data[i + 1] & 0x3Fu);
  } else if ((data[i] & 0xF0u) == 0xE0u) {
    ans = (static_cast<unsigned int>(data[i] & 0x0Fu) << 12) +
          (static_cast<unsigned int>(data[i + 1] & 0x3Fu) << 6) +
          static_cast<unsigned int>(data[i + 2] & 0x3Fu);
  } else {
    ans = (static_cast<unsigned int>(data[i] & 0x07u) << 18) +
          (static_cast<unsigned int>(data[i + 1] & 0x3Fu) << 12) +
          (static_cast<unsigned int>(data[i + 2] & 0x3Fu) << 6) +
          static_cast<unsigned int>(data[i + 3] & 0x3Fu);
  }
  return static_cast<int>(ans);
}

inline bool String::is_small() const {
  const unsigned char category_extract_mask = 0xC0;
  return (data_[max_small_size_] & category_extract_mask) == 0;
}

inline bool String::operator==(const il::String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const unsigned char* p0 = begin();
    const unsigned char* p1 = other.begin();
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool String::has_suffix(const char* data) const {
  il::int_t n = 0;
  while (data[n] != '\0') {
    ++n;
  };
  if (size() < n) {
    return false;
  } else {
    const unsigned char* p0 = end() - n;
    const unsigned char* p1 = reinterpret_cast<const unsigned char*>(data);
    for (il::int_t i = 0; i < n; ++i) {
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

  data_[max_small_size_] = static_cast<unsigned char>(max_small_size_ - n);
}

inline void String::set_large_capacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline const unsigned char* String::data() const {
  if (is_small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline const unsigned char* String::begin() const {
  if (is_small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline unsigned char* String::begin() {
  if (is_small()) {
    return data_;
  } else {
    return large_.data;
  }
}

inline const unsigned char* String::end() const {
  if (is_small()) {
    return data_ + size();
  } else {
    return large_.data + size();
  }
}

inline unsigned char* String::end() {
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
  const il::int_t old_capacity = capacity();
  const bool old_is_small = is_small();
  bool error0;
  bool error1;
  bool error2;
  const il::int_t needed_size = il::safe_sum(old_size, n, il::io, error0);
  il::int_t confort_capacity =
      il::safe_product(static_cast<il::int_t>(2), old_size, il::io, error1);
  confort_capacity = il::safe_sum(confort_capacity, n, il::io, error2);
  if (error0 || error1 || error2) {
    il::abort();
  }
  const il::int_t new_capacity = needed_size <= old_capacity ? old_capacity : il::max(needed_size, confort_capacity);
  unsigned char* old_data = old_is_small ? data_ : large_.data;
  unsigned char* new_data = old_data;
  if (new_capacity > old_capacity) {
    new_data = il::allocate_array<unsigned char>(new_capacity + 1);
    std::memcpy(new_data, old_data, static_cast<std::size_t>(old_size));
  }

  if (new_capacity <= max_small_size_) {
    std::memcpy(data_ + old_size, data, static_cast<std::size_t>(n));
    data_[old_size + n] = static_cast<unsigned char>('\0');
    set_small_size(old_size + n);
  } else {
    std::memcpy(new_data + old_size, data, static_cast<std::size_t>(n));
    new_data[old_size + n] = static_cast<unsigned char>('\0');
    if (new_data != old_data && !old_is_small) {
      il::deallocate(old_data);
    }
    large_.data = new_data;
    large_.size = old_size + n;
    set_large_capacity(new_capacity);
  }
}

inline bool String::valid_code_point(int cp) {
  const unsigned int code_point_max = 0x0010FFFFu;
  const unsigned int lead_surrogate_min = 0x0000D800u;
  const unsigned int lead_surrogate_max = 0x0000DBFFu;
  const unsigned int ucp = static_cast<unsigned int>(cp);
  return ucp <= code_point_max &&
         (ucp < lead_surrogate_min || ucp > lead_surrogate_max);
}

}  // namespace il

#endif  // IL_STRING_H
