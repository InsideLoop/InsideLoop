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
  String& operator=(const char* data);
  String& operator=(const il::String& s);
  String& operator=(il::String&& s);
  ~String();
  il::int_t size() const;
  il::int_t capacity() const;
  bool empty() const;
  bool small() const;
  void reserve(il::int_t r);
  void append(const String& s);
  void append(const char* data);
  void append(const char* data, il::int_t n);
  void append(char c);
  void append(int cp);
  void append(il::int_t n, char c);
  void append(il::int_t n, int cp);
  bool hasSuffix(const char* data) const;
  const char* asCString() const;
  bool operator==(const il::String& other) const;

 private:
  il::int_t small_size() const;
  il::int_t largeCapacity() const;
  void setSmallSize(il::int_t n);
  void setLargeCapacity(il::int_t r);
  const char* begin() const;
  const char* end() const;
  char* begin();
  char* end();
  const char* data() const;
  bool validCodePoint(int cp);
  static il::int_t getSize(const char* data);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeString) - 1);
};

inline String::String() {
  data_[0] = static_cast<unsigned char>('\0');
  setSmallSize(0);
}

inline String::String(const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(n));
    data_[n] = static_cast<unsigned char>('\0');
    setSmallSize(n);
  } else {
    large_.data = il::allocateArray<unsigned char>(n + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(n));
    large_.data[n] = static_cast<unsigned char>('\0');
    large_.size = static_cast<std::size_t>(n);
    setLargeCapacity(n);
  }
}

inline String::String(const char* data) : String{data, String::getSize(data)} {}

inline String::String(const String& s) : String{s.data(), s.size()} {}

inline String::String(String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.asCString(), static_cast<std::size_t>(size) + 1);
    setSmallSize(size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
    s.data_[0] = static_cast<unsigned char>('\0');
    s.setSmallSize(0);
  }
}

inline String& String::operator=(const char* data) {
  const il::int_t size = getSize(data);
  if (size <= max_small_size_) {
    if (!small()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, data, static_cast<std::size_t>(size) + 1);
    setSmallSize(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!small()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocateArray<unsigned char>(size + 1);
      std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
      setLargeCapacity(size);
    }
  }
  return *this;
}

inline String& String::operator=(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    if (!small()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.asCString(), static_cast<std::size_t>(size) + 1);
    setSmallSize(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.asCString(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!small()) {
        il::deallocate(large_.data);
      }
      large_.data = il::allocateArray<unsigned char>(size + 1);
      std::memcpy(large_.data, s.asCString(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
      setLargeCapacity(size);
    }
  }
  return *this;
}

inline String& String::operator=(String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    if (size <= max_small_size_) {
      if (!small()) {
        il::deallocate(large_.data);
      }
      std::memcpy(data_, s.asCString(), static_cast<std::size_t>(size) + 1);
      setSmallSize(size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.data_[0] = static_cast<unsigned char>('\0');
      s.setSmallSize(0);
    }
  }
  return *this;
}

inline String::~String() {
  if (!small()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t String::size() const {
  if (small()) {
    return max_small_size_ - static_cast<il::int_t>(data_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

inline il::int_t String::capacity() const {
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

inline void String::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const bool old_small = small();
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  unsigned char* new_data = il::allocateArray<unsigned char>(r + 1);
  std::memcpy(new_data, asCString(), static_cast<std::size_t>(old_size) + 1);
  if (!old_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  large_.size = old_size;
  setLargeCapacity(r);
}

inline void String::append(const String& s) { append(s.asCString(), s.size()); }

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
  char* data = begin() + old_size;
  data[0] = c;
  data[1] = '\0';
  if (small()) {
    setSmallSize(new_size);
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
  char* data = begin() + old_size;
  for (il::int_t i = 0; i < n; ++i) {
    data[i] = c;
  }
  data[n] = '\0';
  if (small()) {
    setSmallSize(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(int cp) {
  IL_EXPECT_MEDIUM(validCodePoint(cp));

  const unsigned int ucp = static_cast<unsigned int>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00000080u) {
    new_size = old_size + 1;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>(ucp);
    data[1] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00000800u) {
    new_size = old_size + 2;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>((ucp >> 6) | 0x000000C0u);
    data[1] = static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    data[2] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00010000u) {
    new_size = old_size + 3;
    const il::int_t new_capacity =
        il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
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
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>((ucp >> 18) | 0x000000F0u);
    data[1] =
        static_cast<unsigned char>(((ucp >> 12) & 0x0000003Fu) | 0x00000080u);
    data[2] =
        static_cast<unsigned char>(((ucp >> 6) & 0x0000003Fu) | 0x00000080u);
    data[3] = static_cast<unsigned char>((ucp & 0x0000003Fu) | 0x00000080u);
    data[4] = static_cast<unsigned char>('\0');
  }
  if (small()) {
    setSmallSize(new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(il::int_t n, int cp) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(validCodePoint(cp));

  const unsigned int ucp = static_cast<unsigned int>(cp);
  const il::int_t old_size = size();
  il::int_t new_size;
  if (ucp < 0x00000080u) {
    new_size = old_size + n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    const unsigned char cu0 = static_cast<unsigned char>(ucp);
    for (il::int_t i = 0; i < n; ++i) {
      data[i] = cu0;
    }
    data[n] = static_cast<unsigned char>('\0');
  } else if (ucp < 0x00000800u) {
    new_size = old_size + 2 * n;
    const il::int_t new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
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
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
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
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
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
  if (small()) {
    setSmallSize(new_size);
  } else {
    large_.size = new_size;
  }
}

inline const char* String::asCString() const {
  if (small()) {
    return reinterpret_cast<const char*>(data_);
  } else {
    return reinterpret_cast<const char*>(large_.data);
  }
}

inline bool String::empty() const { return size() == 0; }

inline bool String::small() const {
  const unsigned char category_extract_mask = 0xC0;
  return (data_[max_small_size_] & category_extract_mask) == 0;
}

inline bool String::operator==(const il::String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const unsigned char* p0 = reinterpret_cast<const unsigned char*>(begin());
    const unsigned char* p1 =
        reinterpret_cast<const unsigned char*>(other.begin());
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool String::hasSuffix(const char* data) const {
  il::int_t n = 0;
  while (data[n] != '\0') {
    ++n;
  };
  if (size() < n) {
    return false;
  } else {
    const unsigned char* p0 = reinterpret_cast<const unsigned char*>(end()) - n;
    const unsigned char* p1 = reinterpret_cast<const unsigned char*>(data);
    for (il::int_t i = 0; i < n; ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline il::int_t String::getSize(const char* data) {
  il::int_t i = 0;
  while (data[i] != '\0') {
    ++i;
  }
  return i;
}

inline il::int_t String::small_size() const {
  return max_small_size_ - static_cast<il::int_t>(data_[max_small_size_]);
}

inline il::int_t String::largeCapacity() const {
  constexpr unsigned char category_extract_mask = 0xC0;
  constexpr std::size_t capacity_extract_mask =
      ~(static_cast<std::size_t>(category_extract_mask)
        << ((sizeof(std::size_t) - 1) * 8));
  return static_cast<il::int_t>(large_.capacity & capacity_extract_mask);
}

inline void String::setSmallSize(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  data_[max_small_size_] = static_cast<unsigned char>(max_small_size_ - n);
}

inline void String::setLargeCapacity(il::int_t r) {
  large_.capacity =
      static_cast<std::size_t>(r) |
      (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
}

inline const char* String::data() const {
  if (small()) {
    return reinterpret_cast<const char*>(data_);
  } else {
    return reinterpret_cast<const char*>(large_.data);
  }
}

inline const char* String::begin() const {
  if (small()) {
    return reinterpret_cast<const char*>(data_);
  } else {
    return reinterpret_cast<const char*>(large_.data);
  }
}

inline char* String::begin() {
  if (small()) {
    return reinterpret_cast<char*>(data_);
  } else {
    return reinterpret_cast<char*>(large_.data);
  }
}

inline const char* String::end() const {
  if (small()) {
    return reinterpret_cast<const char*>(data_) + size();
  } else {
    return reinterpret_cast<const char*>(large_.data) + size();
  }
}

inline char* String::end() {
  if (small()) {
    return reinterpret_cast<char*>(data_) + size();
  } else {
    return reinterpret_cast<char*>(large_.data) + size();
  }
}

inline void String::append(const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const bool old_small = small();
  const il::int_t old_size = old_small ? small_size() : large_.size;
  const il::int_t old_capacity = old_small ? max_small_size_ : largeCapacity();
  unsigned char* old_data = old_small ? data_ : large_.data;

  const std::size_t u_needed_size =
      static_cast<std::size_t>(old_size) + static_cast<std::size_t>(n);
  il::int_t new_capacity;
  bool needs_new_buffer;
  if (u_needed_size <= static_cast<std::size_t>(old_capacity)) {
    new_capacity = old_capacity;
    needs_new_buffer = false;
  } else {
    const std::size_t u_confort_capacity =
        2 * static_cast<std::size_t>(old_size);
    const std::size_t u_new_capacity =
        il::max(u_needed_size, u_confort_capacity);
    constexpr std::size_t sentinel_integer = static_cast<std::size_t>(1)
                                             << (sizeof(std::size_t) * 8 - 1);
    if (u_new_capacity + 1 >= sentinel_integer) {
      il::abort();
    }
    new_capacity = static_cast<il::int_t>(u_new_capacity);
    needs_new_buffer = true;
  }

  unsigned char* new_data;
  if (needs_new_buffer) {
    new_data = il::allocateArray<unsigned char>(new_capacity + 1);
    std::memcpy(new_data, old_data, static_cast<std::size_t>(old_size));
  } else {
    new_data = old_data;
  }
  std::memcpy(new_data + old_size, data, static_cast<std::size_t>(n) + 1);
  if (new_capacity <= max_small_size_) {
    setSmallSize(old_size + n);
  } else {
    if (needs_new_buffer && !old_small) {
      il::deallocate(old_data);
    }
    large_.data = new_data;
    large_.size = old_size + n;
    setLargeCapacity(new_capacity);
  }
}

inline bool String::validCodePoint(int cp) {
  const unsigned int code_point_max = 0x0010FFFFu;
  const unsigned int lead_surrogate_min = 0x0000D800u;
  const unsigned int lead_surrogate_max = 0x0000DBFFu;
  const unsigned int ucp = static_cast<unsigned int>(cp);
  return ucp <= code_point_max &&
         (ucp < lead_surrogate_min || ucp > lead_surrogate_max);
}

}  // namespace il

#endif  // IL_STRING_H
