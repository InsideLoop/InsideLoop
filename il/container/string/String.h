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
#include <iostream>

#include <il/base.h>
#include <il/core/memory/allocate.h>

namespace il {

// LargeString is defined this way (64-bit system, little endian as on Linux
// and Windows)
//
// - For small string optimization: 24 chars which makes 24 bytes
//   The first 22 chars can be used to store the string. Another char is used
//   to store the termination character '\0'. The last char is used to store
//   the length of the string and its type: ascii, utf8, wtf8 or byte
// - For large strings: 1 pointer and 2 std::size_t which make 24 bytes
//
// The last bit (the most significant ones as we are on a little endian
// system), is used to know if the string is a small string or no (0 for small
// string and 1 for large string).
//
// The 2 bits before are used to know the kind of the string. We use (little
// endian):
// 00: ascii
// 10: utf8
// 01: wtf8
// 11: byte

constexpr il::int_t mySizeCString(const char* s, int i) {
  return *s == '\0' ? i : mySizeCString(s + 1, i + 1);
}

constexpr bool myIsAscii(const char* s) {
  return (*s == '\0') ||
         ((static_cast<unsigned char>(*s) < static_cast<unsigned char>(128))
              ? myIsAscii(s + 1)
              : false);
}

inline il::int_t cstringSizeType(const char* s, il::io_t, bool& is_ascii) {
  is_ascii = true;
  il::int_t n = 0;
  while (s[n] != '\0') {
    if ((static_cast<unsigned char>(s[n]) & 0x80_uchar) != 0x00_uchar) {
      is_ascii = false;
    }
    ++n;
  }
  return n;
}

inline il::int_t size(const char* data) {
  il::int_t i = 0;
  while (data[i] != '\0') {
    ++i;
  }
  return i;
}

inline constexpr bool isAscii(const char* data) {
  return data[0] == '\0' ? true
                         : ((static_cast<unsigned char>(data[0]) &
                             0x80_uchar) == 0x00_uchar) &&
                               isAscii(data + 1);
}

// template <il::int_t m>
// inline bool isAscii(const char (&data)[m]) {
//  bool ans = true;
//  for (il::int_t i = 0; i < m - 1; ++i) {
//    if ((static_cast<unsigned char>(data[i]) & 0x80_uchar) != 0x00_uchar) {
//      ans = false;
//    }
//  }
//  return ans;
//}

inline constexpr bool auxIsUtf8(const char* s, int nbBytes, int pos,
                                bool surrogate) {
  return (pos == 0)
             // If it starts with the null character, the string is valid
             ? ((*s == 0x00)
                    ? true
                    // Otherwise, let's check if is starts with an ASCII
                    // character
                    : ((*s & 0x80) == 0
                           // In this case, check the rest of the string
                           ? auxIsUtf8(s + 1, 0, 0, false)
                           // Otherwise, it might start with a 2-byte sequence
                           : ((*s & 0xD0) == 0xB0
                                  // In this case, check the rest of the string
                                  ? auxIsUtf8(s + 1, 2, 1, false)
                                  // Otherwise, it might start with a 3-byte
                                  // sequence
                                  : ((*s & 0xF0) == 0xD0
                                         // In this case, check the rest of the
                                         // string
                                         ? auxIsUtf8(
                                               s + 1, 3, 1,
                                               (static_cast<unsigned char>(
                                                    *s) == 0xED))
                                         // Otherwise, it might start with a
                                         // 4-byte sequence
                                         : ((*s & 0xF8) == 0xF0
                                                ? auxIsUtf8(s + 1, 4, 1, false)
                                                : false)))))
             // In the case where we are scanning the second byte of a multibyte
             // sequence
             : ((pos == 1)
                    ? ((*s & 0xC0) == 0x80
                           ? (nbBytes == 2 ? ((*s & 0xA0) != 0xA0)
                                           : auxIsUtf8(s + 1, nbBytes, pos + 1,
                                                       surrogate))
                           : false)
                    // In the case where we are scanning the third byte of a
                    // multibyte sequence
                    : ((pos == 2)
                           ? ((*s & 0xC0) == 0x80
                                  ? (nbBytes == 3
                                         ? true
                                         : auxIsUtf8(s + 1, nbBytes, pos + 1,
                                                     surrogate))
                                  : false)
                           // In the case where we are scanning the
                           // fourth byte of a multibyte sequence
                           : ((pos == 3) ? ((*s & 0xC0) == 0x80) : false)));
}

inline constexpr bool isUtf8(const char* s) {
  return auxIsUtf8(s, 0, 0, false);
}

enum class StringType : unsigned char {
  Ascii = 0x00_uchar,
  Utf8 = 0x20_uchar,
  Wtf8 = 0x40_uchar,
  Bytes = 0x60_uchar
};

inline il::StringType joinType(il::StringType t0, il::StringType t1) {
  return static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1)));
}

inline il::StringType joinType(il::StringType t0, il::StringType t1,
                               il::StringType t2) {
  return static_cast<il::StringType>(il::max(static_cast<unsigned char>(t0),
                                             static_cast<unsigned char>(t1),
                                             static_cast<unsigned char>(t2)));
}

inline il::StringType joinType(il::StringType t0, il::StringType t1,
                               il::StringType t2, il::StringType t3) {
  return static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1),
              static_cast<unsigned char>(t2), static_cast<unsigned char>(t3)));
}

class String {
 private:
  struct LargeString {
    char* data;
    il::int_t size;
    std::size_t capacity;
  };
  union {
    char data_[sizeof(LargeString)];
    LargeString large_;
  };

 public:
  String();
  template <il::int_t m>
  String(const char (&data)[m]);
  template <il::int_t m>
  String(il::StringType, const char (&data)[m]);
  explicit String(const char* data, il::int_t n);
  explicit String(il::StringType type, const char* data, il::int_t n);
  explicit String(il::unsafe_t, il::int_t n);
  String(const il::String& s);
  String(il::String&& s);
  String& operator=(const char* data);
  String& operator=(const il::String& s);
  String& operator=(il::String&& s);
  ~String();
  il::int_t size() const;
  il::int_t capacity() const;
  bool isEmpty() const;
  bool isSmall() const;
  il::StringType type() const;
  bool isRuneBoundary(il::int_t i) const;

  void reserve(il::int_t r);

  void setSafe(il::StringType type, il::int_t n);

  template <il::int_t m>
  void append(const char (&s0)[m]);
  void append(const String& s);
  void append(const String& s0, const String& s1);
  template <il::int_t m>
  void append(const String& s0, const char (&s1)[m]);
  template <il::int_t m>
  void append(const char (&s0)[m], const String& s1);
  void append(const String& s0, const String& s1, const String& s2);
  template <il::int_t m>
  void append(const char (&s0)[m], const String& s1, const String& s2);
  template <il::int_t m>
  void append(const String& s0, const char (&s1)[m], const String& s2);
  template <il::int_t m>
  void append(const String& s0, const String& s1, const char (&s2)[m]);
  template <il::int_t m0, il::int_t m2>
  void append(const char (&s0)[m0], const String& s1, const char (&s2)[m2]);

  void append(char c);
  void append(int rune);
  void append(il::int_t n, char c);
  void append(il::int_t n, int rune);
  void append(const char* data, il::int_t n);
  void append(il::StringType type, const char* data, il::int_t n);

  il::String substring(il::int_t i0, il::int_t i1) const;
  il::String prefix(il::int_t n) const;
  il::String suffix(il::int_t n) const;

  void clear();

  bool endsWith(const char* data) const;

  const char* asCString() const;
  bool operator==(const il::String& other) const;
  bool isEqual(const char* data) const;
  const char* data() const;
  const char* begin() const;
  const char* end() const;
  char* data();
  char* begin();
  char* end();
  //  void truncate(il::int_t n);
  //  void setRaw();
  //  il::String split(il::int_t n);
  //  void resize(il::unsafe_t, il::int_t n);
  friend std::ostream& operator<<(std::ostream& os, const String& s) {
    return os << s.data();
  }

 private:
  il::int_t smallSize() const;
  il::int_t largeCapacity() const;
  void setSmall(il::StringType type, il::int_t n);
  void setLarge(il::StringType type, il::int_t n, il::int_t r);
  bool validRune(int rune);
  il::int_t static constexpr sizeCString(const char* data);
  il::int_t static nextCapacity(il::int_t n);
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeString) - 2);
};

inline String::String() {
  data_[0] = '\0';
  setSmall(il::StringType::Ascii, 0);
}

template <il::int_t m>
inline String::String(const char (&data)[m]) {
  const il::int_t n = m - 1;
  if (n <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(m));
    setSmall(il::StringType::Utf8, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(m));
    setLarge(il::StringType::Utf8, n, r);
  }
}

template <il::int_t m>
inline String::String(il::StringType type, const char (&data)[m]) {
  const il::int_t n = m - 1;
  const bool is_ascii = (type == il::StringType::Ascii);
  if (n <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(m));
    setSmall(is_ascii ? il::StringType::Ascii : il::StringType::Utf8, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(m));
    setLarge(is_ascii ? il::StringType::Ascii : il::StringType::Utf8, n, r);
  }
}

inline String::String(const char* data, il::int_t n)
    : String{il::StringType::Bytes, data, n} {}

inline String::String(il::StringType type, const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    std::memcpy(data_, data, static_cast<std::size_t>(n));
    data_[n] = '\0';
    setSmall(type, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    std::memcpy(large_.data, data, static_cast<std::size_t>(n));
    large_.data[n] = '\0';
    setLarge(type, n, r);
  }
}

inline String::String(il::unsafe_t, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  if (n <= max_small_size_) {
    setSmall(il::StringType::Ascii, n);
  } else {
    const il::int_t r = nextCapacity(n);
    large_.data = il::allocateArray<char>(r + 1);
    setLarge(il::StringType::Ascii, n, r);
  }
}

inline String::String(const String& s) : String{s.type(), s.data(), s.size()} {}

inline String::String(String&& s) {
  const il::int_t size = s.size();
  const il::StringType type = s.type();
  if (size <= max_small_size_) {
    std::memcpy(data_, s.asCString(), static_cast<std::size_t>(size) + 1);
    setSmall(type, size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity = s.large_.capacity;
    s.data_[0] = '\0';
    s.setSmall(il::StringType::Ascii, 0);
  }
}

inline String& String::operator=(const char* data) {
  bool is_ascii;
  const il::int_t size = il::cstringSizeType(data, il::io, is_ascii);
  const il::StringType type =
      is_ascii ? il::StringType::Ascii : il::StringType::Utf8;
  if (size <= max_small_size_) {
    if (!isSmall()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, data, static_cast<std::size_t>(size) + 1);
    setSmall(type, size);
  } else {
    const il::int_t old_r = capacity();
    if (size <= old_r) {
      std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
      setLarge(type, size, old_r);
      large_.size = size;
    } else {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      const il::int_t new_r = nextCapacity(size);
      large_.data = il::allocateArray<char>(new_r + 1);
      std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
      setLarge(type, size, new_r);
    }
  }
  return *this;
}

inline String& String::operator=(const String& s) {
  const il::int_t size = s.size();
  const il::StringType type = s.type();
  if (size <= max_small_size_) {
    if (!isSmall()) {
      il::deallocate(large_.data);
    }
    std::memcpy(data_, s.asCString(), static_cast<std::size_t>(size) + 1);
    setSmall(type, size);
  } else {
    const il::int_t old_r = capacity();
    if (size <= old_r) {
      std::memcpy(large_.data, s.asCString(),
                  static_cast<std::size_t>(size) + 1);
      setLarge(type, size, old_r);
    } else {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      const il::int_t new_r = nextCapacity(size);
      large_.data = il::allocateArray<char>(new_r + 1);
      std::memcpy(large_.data, s.asCString(),
                  static_cast<std::size_t>(size) + 1);
      setLarge(type, size, new_r);
    }
  }
  return *this;
}

inline String& String::operator=(String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    const il::StringType type = s.type();
    if (size <= max_small_size_) {
      if (!isSmall()) {
        il::deallocate(large_.data);
      }
      std::memcpy(data_, s.asCString(), static_cast<std::size_t>(size) + 1);
      setSmall(type, size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity = s.large_.capacity;
      s.data_[0] = '\0';
      s.setSmall(il::StringType::Ascii, 0);
    }
  }
  return *this;
}

inline String::~String() {
  if (!isSmall()) {
    il::deallocate(large_.data);
  }
}

inline il::int_t String::size() const {
  if (isSmall()) {
    return static_cast<il::int_t>(
        static_cast<unsigned char>(data_[max_small_size_ + 1]) & 0x1F_uchar);
  } else {
    return large_.size;
  }
}

inline il::int_t String::capacity() const {
  if (isSmall()) {
    return max_small_size_;
  } else {
    return static_cast<il::int_t>(
        (static_cast<std::size_t>(large_.capacity) &
         ~(static_cast<std::size_t>(0xD0) << ((sizeof(std::size_t) - 1) * 8)))
        << 3);
  }
}

inline bool String::isRuneBoundary(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));

  const char* data = this->data();
  const unsigned char c = static_cast<unsigned char>(data[i]);
  return c < 0x80_uchar || ((c & 0xC0_uchar) != 0x80_uchar);
}

inline void String::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const bool old_small = isSmall();
  const il::StringType type = this->type();
  const il::int_t size = this->size();
  const il::int_t new_r = nextCapacity(r);
  char* new_data = il::allocateArray<char>(new_r + 1);
  std::memcpy(new_data, asCString(), static_cast<std::size_t>(size) + 1);
  if (!old_small) {
    il::deallocate(large_.data);
  }
  large_.data = new_data;
  setLarge(type, size, new_r);
}

inline void String::setSafe(il::StringType type, il::int_t n) {
  if (isSmall()) {
    data_[n] = '\0';
    setSmall(type, n);
  } else {
    large_.data[n] = '\0';
    large_.size = n;
  }
}

inline void String::clear() {
  if (isSmall()) {
    data_[0] = '\0';
    setSmall(il::StringType::Ascii, 0);
  } else {
    large_.data[0] = '\0';
    setLarge(il::StringType::Ascii, 0, capacity());
  }
}

inline void String::append(il::StringType type, const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::StringType old_type = this->type();
  const il::StringType new_type =
      type == il::StringType::Ascii
          ? old_type
          : (old_type == il::StringType::Ascii ? il::StringType::Utf8
                                               : old_type);

  const bool old_small = isSmall();
  const il::int_t old_size = old_small ? smallSize() : large_.size;
  const il::int_t old_capacity = old_small ? max_small_size_ : largeCapacity();
  char* old_data = old_small ? data_ : large_.data;

  const il::int_t needed_size = old_size + n;
  il::int_t new_capacity;
  bool needs_new_buffer;
  if (needed_size <= old_capacity) {
    new_capacity = old_capacity;
    needs_new_buffer = false;
  } else {
    new_capacity = nextCapacity(2 * needed_size);
    needs_new_buffer = true;
  }

  char* new_data;
  if (needs_new_buffer) {
    new_data = il::allocateArray<char>(new_capacity + 1);
    std::memcpy(new_data, old_data, static_cast<std::size_t>(old_size));
  } else {
    new_data = old_data;
  }
  std::memcpy(new_data + old_size, data, static_cast<std::size_t>(n) + 1);
  if (new_capacity <= max_small_size_) {
    setSmall(new_type, old_size + n);
  } else {
    if (needs_new_buffer && !old_small) {
      il::deallocate(old_data);
    }
    large_.data = new_data;
    setLarge(new_type, old_size + n, new_capacity);
  }
}

inline il::String String::substring(il::int_t i0, il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i0 <= i1);

  if (type() == il::StringType::Utf8) {
    IL_EXPECT_MEDIUM(isRuneBoundary(i0));
    IL_EXPECT_MEDIUM(isRuneBoundary(i1));
  }
  return il::String{type(), data_ + i0, i1 - i0};
}

inline il::String String::suffix(il::int_t n) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(size()));

  if (type() == il::StringType::Utf8) {
    IL_EXPECT_MEDIUM(isRuneBoundary(size() - n));
  }
  return il::String{type(), data_ + size() - n, n};
}

inline String String::prefix(il::int_t n) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(size()));

  if (type() == il::StringType::Utf8) {
    IL_EXPECT_MEDIUM(isRuneBoundary(n));
  }
  return il::String{type(), data_, n};
}

template <il::int_t m>
inline void String::append(const char (&s0)[m]) {
  append(il::StringType::Utf8, s0, m - 1);
}

inline void String::append(const String& s) {
  append(s.type(), s.data(), s.size());
}

inline void String::append(const String& s0, const String& s1) {
  append(s0.type(), s0.data(), s0.size());
  append(s1.type(), s1.data(), s1.size());
}

template <il::int_t m>
inline void String::append(const String& s0, const char (&s1)[m]) {
  append(s0.type(), s0.data(), s0.size());
  append(il::StringType::Utf8, s1, m - 1);
}

template <il::int_t m>
inline void String::append(const char (&s0)[m], const String& s1) {
  const il::int_t old_n = size();
  const il::int_t n0 = m - 1;
  const il::int_t n1 = s1.size();
  const il::StringType t = joinType(type(), il::StringType::Utf8, s1.type());
  reserve(old_n + n0 + n1);
  char* p = data();
  std::memcpy(p + old_n, s0, n0);
  std::memcpy(p + old_n + n0, s1.data(), n1 + 1);
  setSafe(t, old_n + n0 + n1);
}

inline void String::append(const String& s0, const String& s1,
                           const String& s2) {
  const il::int_t old_n = size();
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::StringType t = joinType(type(), s0.type(), s1.type(), s2.type());
  reserve(old_n + n0 + n1 + n2);
  char* p = data();
  std::memcpy(p + old_n, s0.data(), n0);
  std::memcpy(p + old_n + n0, s1.data(), n1);
  std::memcpy(p + old_n + n0 + n1, s2.data(), n2 + 1);
  setSafe(t, old_n + n0 + n1 + n2);
}

template <il::int_t m>
inline void String::append(const char (&s0)[m], const String& s1,
                           const String& s2) {
  const il::int_t old_n = size();
  const il::int_t n0 = m - 1;
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::StringType t =
      joinType(type(), il::StringType::Utf8, s1.type(), s2.type());
  reserve(old_n + n0 + n1 + n2);
  char* p = data();
  std::memcpy(p + old_n, s0, n0);
  std::memcpy(p + old_n + n0, s1.data(), n1);
  std::memcpy(p + old_n + n0 + n1, s2.data(), n2 + 1);
  setSafe(t, old_n + n0 + n1 + n2);
}

template <il::int_t m>
inline void String::append(const String& s0, const char (&s1)[m],
                           const String& s2) {
  const il::int_t old_n = size();
  const il::int_t n0 = s0.size();
  const il::int_t n1 = m - 1;
  const il::int_t n2 = s2.size();
  const il::StringType t =
      joinType(type(), s0.type(), il::StringType::Utf8, s2.type());
  reserve(old_n + n0 + n1 + n2);
  char* p = data();
  std::memcpy(p + old_n, s0.data(), n0);
  std::memcpy(p + old_n + n0, s1, n1);
  std::memcpy(p + old_n + n0 + n1, s2.data(), n2 + 1);
  setSafe(t, old_n + n0 + n1 + n2);
}

template <il::int_t m>
inline void String::append(const String& s0, const String& s1,
                           const char (&s2)[m]) {
  const il::int_t old_n = size();
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = m - 1;
  const il::StringType t =
      joinType(type(), s0.type(), s1.type(), il::StringType::Utf8);
  reserve(old_n + n0 + n1 + n2);
  char* p = data();
  std::memcpy(p + old_n, s0.data(), n0);
  std::memcpy(p + old_n + n0, s1.data(), n1);
  std::memcpy(p + old_n + n0 + n1, s2, m);
  setSafe(t, old_n + n0 + n1 + n2);
}

template <il::int_t m0, il::int_t m2>
inline void String::append(const char (&s0)[m0], const String& s1,
                           const char (&s2)[m2]) {
  const il::int_t old_n = size();
  const il::int_t n0 = m0 - 1;
  const il::int_t n1 = s1.size();
  const il::int_t n2 = m2 - 1;
  const il::StringType t =
      joinType(type(), il::StringType::Utf8, s1.type(), il::StringType::Utf8);
  reserve(old_n + n0 + n1 + n2);
  char* p = data();
  std::memcpy(p + old_n, s0, n0);
  std::memcpy(p + old_n + n0, s1.data(), n1);
  std::memcpy(p + old_n + n0 + n1, s2, m2);
  setSafe(t, old_n + n0 + n1 + n2);
}

inline void String::append(const char* data, il::int_t n) {
  append(il::StringType::Bytes, data, n);
}

void append(const char* data, il::int_t n);

inline void String::append(char c) {
  IL_EXPECT_MEDIUM(static_cast<unsigned char>(c) < 128);

  const il::int_t old_size = size();
  const il::int_t new_size = old_size + 1;
  const il::int_t new_capacity = il::max(new_size, 2 * old_size);
  reserve(new_capacity);
  char* data = begin() + old_size;
  data[0] = c;
  data[1] = '\0';
  if (isSmall()) {
    setSmall(type(), new_size);
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
  if (isSmall()) {
    setSmall(type(), new_size);
  } else {
    large_.size = new_size;
  }
}

inline void String::append(int rune) {
  IL_EXPECT_MEDIUM(validRune(rune));

  const unsigned int urune = static_cast<unsigned int>(rune);
  const il::int_t old_size = size();
  il::int_t new_size;
  il::int_t new_capacity;
  if (urune < 0x00000080u) {
    new_size = old_size + 1;
    new_capacity = il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>(urune);
    data[1] = '\0';
  } else if (urune < 0x00000800u) {
    new_size = old_size + 2;
    new_capacity = il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>((urune >> 6) | 0x000000C0u);
    data[1] = static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    data[2] = '\0';
  } else if (urune < 0x00010000u) {
    new_size = old_size + 3;
    new_capacity = il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>((urune >> 12) | 0x000000E0u);
    data[1] =
        static_cast<unsigned char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
    data[2] = static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    data[3] = '\0';
  } else {
    new_size = old_size + 4;
    new_capacity = il::max(new_size, il::min(max_small_size_, 2 * old_size));
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    data[0] = static_cast<unsigned char>((urune >> 18) | 0x000000F0u);
    data[1] =
        static_cast<unsigned char>(((urune >> 12) & 0x0000003Fu) | 0x00000080u);
    data[2] =
        static_cast<unsigned char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
    data[3] = static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    data[4] = '\0';
  }
  const il::StringType old_type = type();
  const il::StringType new_type =
      rune < 128 ? old_type
                 : (old_type == il::StringType::Ascii ? il::StringType::Utf8
                                                      : old_type);
  if (isSmall()) {
    setSmall(new_type, new_size);
  } else {
    setLarge(new_type, new_size, new_capacity);
    large_.size = new_size;
  }
}

inline void String::append(il::int_t n, int rune) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_FAST(validRune(rune));

  const unsigned int urune = static_cast<unsigned int>(rune);
  const il::int_t old_size = size();
  il::int_t new_size;
  il::int_t new_capacity;
  if (urune < 0x00000080u) {
    new_size = old_size + n;
    new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    const unsigned char cu0 = static_cast<unsigned char>(urune);
    for (il::int_t i = 0; i < n; ++i) {
      data[i] = cu0;
    }
    data[n] = '\0';
  } else if (urune < 0x00000800u) {
    new_size = old_size + 2 * n;
    new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    const unsigned char cu0 =
        static_cast<unsigned char>((urune >> 6) | 0x000000C0u);
    const unsigned char cu1 =
        static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[2 * i] = cu0;
      data[2 * i + 1] = cu1;
    }
    data[2 * n] = '\0';
  } else if (urune < 0x00010000u) {
    new_size = old_size + 3 * n;
    new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    const unsigned char cu0 =
        static_cast<unsigned char>((urune >> 12) | 0x000000E0u);
    const unsigned char cu1 =
        static_cast<unsigned char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
    const unsigned char cu2 =
        static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[3 * i] = cu0;
      data[3 * i + 1] = cu1;
      data[3 * i + 2] = cu2;
    }
    data[3 * n] = '\0';
  } else {
    new_size = old_size + 4 * n;
    new_capacity = il::max(new_size, 2 * old_size);
    reserve(new_capacity);
    unsigned char* data = reinterpret_cast<unsigned char*>(end());
    const unsigned char cu0 =
        static_cast<unsigned char>((urune >> 18) | 0x000000F0u);
    const unsigned char cu1 =
        static_cast<unsigned char>(((urune >> 12) & 0x0000003Fu) | 0x00000080u);
    const unsigned char cu2 =
        static_cast<unsigned char>(((urune >> 6) & 0x0000003Fu) | 0x00000080u);
    const unsigned char cu3 =
        static_cast<unsigned char>((urune & 0x0000003Fu) | 0x00000080u);
    for (il::int_t i = 0; i < n; ++i) {
      data[4 * i] = cu0;
      data[4 * i + 1] = cu1;
      data[4 * i + 2] = cu2;
      data[4 * i + 3] = cu3;
    }
    data[4 * n] = '\0';
  }
  const il::StringType old_type = type();
  const il::StringType new_type =
      rune < 128 ? old_type
                 : (old_type == il::StringType::Ascii ? il::StringType::Utf8
                                                      : old_type);
  if (isSmall()) {
    setSmall(new_type, new_size);
  } else {
    setLarge(new_type, new_size, new_capacity);
    large_.size = new_size;
  }
}

inline const char* String::asCString() const {
  return isSmall() ? data_ : large_.data;
}

inline bool String::isEmpty() const { return size() == 0; }

inline bool String::isSmall() const {
  return (static_cast<unsigned char>(data_[max_small_size_ + 1]) &
          0x80_uchar) == 0x00_uchar;
}

inline il::StringType String::type() const {
  return static_cast<il::StringType>(
      static_cast<unsigned char>(data_[max_small_size_ + 1]) & 0x60_uchar);
}

inline bool String::operator==(const il::String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const char* p0 = begin();
    const char* p1 = other.begin();
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

inline bool String::isEqual(const char* data) const {
  bool ans = true;
  const char* p = begin();
  il::int_t i = 0;
  while (ans && i < size()) {
    if (p[i] != data[i] || data[i] == '\0') {
      ans = false;
    }
    ++i;
  }
  return ans;
}

inline bool String::endsWith(const char* data) const {
  il::int_t n = 0;
  while (data[n] != '\0') {
    ++n;
  };
  if (size() < n) {
    return false;
  } else {
    const char* p0 = end() - n;
    for (il::int_t i = 0; i < n; ++i) {
      if (p0[i] != data[i]) {
        return false;
      }
    }
    return true;
  }
}

inline il::int_t String::smallSize() const {
  return data_[max_small_size_ + 1] & static_cast<unsigned char>(0x1F);
}

inline il::int_t String::largeCapacity() const {
  constexpr unsigned char category_extract_mask = 0xD0_uchar;
  constexpr std::size_t capacity_extract_mask =
      ~(static_cast<std::size_t>(category_extract_mask)
        << ((sizeof(std::size_t) - 1) * 8));
  return static_cast<il::int_t>(
      (static_cast<std::size_t>(large_.capacity) & capacity_extract_mask) << 3);
}

inline void String::setSmall(il::StringType type, il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  unsigned char value = static_cast<unsigned char>(n);
  value |= static_cast<unsigned char>(type);
  data_[max_small_size_ + 1] = value;
}

inline void String::setLarge(il::StringType type, il::int_t n, il::int_t r) {
  IL_EXPECT_MEDIUM(n >= 0);
  IL_EXPECT_MEDIUM(n <= r);

  large_.size = n;
  large_.capacity =
      (static_cast<std::size_t>(r) >> 3) |
      (static_cast<std::size_t>(static_cast<unsigned char>(type))
       << ((sizeof(std::size_t) - 1) * 8)) |
      (static_cast<std::size_t>(0x80_uchar) << ((sizeof(std::size_t) - 1) * 8));
}

inline const char* String::data() const {
  return isSmall() ? data_ : large_.data;
}

inline char* String::data() { return isSmall() ? data_ : large_.data; }

inline const char* String::begin() const {
  return isSmall() ? data_ : large_.data;
}

inline char* String::begin() { return isSmall() ? data_ : large_.data; }

inline const char* String::end() const {
  return (isSmall() ? data_ : large_.data) + size();
}

inline char* String::end() {
  return (isSmall() ? data_ : large_.data) + size();
}

inline bool String::validRune(int rune) {
  const unsigned int code_point_max = 0x0010FFFFu;
  const unsigned int lead_surrogate_min = 0x0000D800u;
  const unsigned int lead_surrogate_max = 0x0000DBFFu;
  const unsigned int urune = static_cast<unsigned int>(rune);
  return urune <= code_point_max &&
         (urune < lead_surrogate_min || urune > lead_surrogate_max);
}

inline il::int_t constexpr String::sizeCString(const char* data) {
  return data[0] == '\0' ? 0 : sizeCString(data + 1) + 1;
}

// Return the next integer modulo 4
inline il::int_t String::nextCapacity(il::int_t r) {
  return static_cast<il::int_t>(
      (static_cast<std::size_t>(r) + static_cast<std::size_t>(3)) &
      ~static_cast<std::size_t>(3));
}

inline il::String join(const il::String& s0, const il::String& s1) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::StringType t0 = s0.type();
  const il::StringType t1 = s1.type();
  il::String ans{il::unsafe, n0 + n1};
  char* data = ans.data();
  std::memcpy(data, s0.data(), static_cast<std::size_t>(n0));
  std::memcpy(data + n0, s1.data(), static_cast<std::size_t>(n1 + 1));
  const il::StringType t = static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1)));
  ans.setSafe(t, n0 + n1);
  return ans;
}

template <il::int_t m>
inline il::String join(const il::String& s0, const char (&s1)[m]) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = m - 1;
  const il::StringType t0 = s0.type();
  const il::StringType t1 = il::StringType::Utf8;
  il::String ans{il::unsafe, n0 + n1};
  char* data = ans.data();
  std::memcpy(data, s0.data(), static_cast<std::size_t>(n0));
  std::memcpy(data + n0, s1, static_cast<std::size_t>(m));
  const il::StringType t = static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1)));
  ans.setSafe(t, n0 + n1);
  return ans;
}

template <il::int_t m>
inline il::String join(const char (&s0)[m], const il::String& s1) {
  const il::int_t n0 = m - 1;
  const il::int_t n1 = s1.size();
  const il::StringType t0 = il::StringType::Utf8;
  const il::StringType t1 = s1.type();
  il::String ans{il::unsafe, n0 + n1};
  char* data = ans.data();
  std::memcpy(data, s0, static_cast<std::size_t>(n0));
  std::memcpy(data + n0, s1.data(), static_cast<std::size_t>(n1 + 1));
  const il::StringType t = static_cast<il::StringType>(
      il::max(static_cast<unsigned char>(t0), static_cast<unsigned char>(t1)));
  ans.setSafe(t, n0 + n1);
  return ans;
}

inline il::String join(const il::String& s0, const il::String& s1,
                       const il::String& s2) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::StringType t = il::joinType(s0.type(), s1.type(), s2.type());
  il::String ans{il::unsafe, n0 + n1 + n2};
  char* p = ans.data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2.data(), n2 + 1);
  ans.setSafe(t, n0 + n1 + n2);
  return ans;
}

template <il::int_t m>
inline il::String join(const char (&s0)[m], const il::String& s1,
                       const il::String& s2) {
  const il::int_t n0 = m - 1;
  const il::int_t n1 = s1.size();
  const il::int_t n2 = s2.size();
  const il::StringType t =
      il::joinType(il::StringType::Utf8, s1.type(), s2.type());
  il::String ans{il::unsafe, n0 + n1 + n2};
  char* p = ans.data();
  std::memcpy(p, s0, n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2.data(), n2 + 1);
  ans.setSafe(t, n0 + n1 + n2);
  return ans;
}

template <il::int_t m>
inline il::String join(const il::String& s0, const char (&s1)[m],
                       const il::String& s2) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = m - 1;
  const il::int_t n2 = s2.size();
  const il::StringType t =
      il::joinType(s0.type(), il::StringType::Utf8, s2.type());
  il::String ans{il::unsafe, n0 + n1 + n2};
  char* p = ans.data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1, n1);
  std::memcpy(p + n0 + n1, s2.data(), n2 + 1);
  ans.setSafe(t, n0 + n1 + n2);
  return ans;
}

template <il::int_t m>
inline il::String join(const il::String& s0, const il::String& s1,
                       const char (&s2)[m]) {
  const il::int_t n0 = s0.size();
  const il::int_t n1 = s1.size();
  const il::int_t n2 = m - 1;
  const il::StringType t =
      il::joinType(s0.type(), s1.type(), il::StringType::Utf8);
  il::String ans{il::unsafe, n0 + n1 + n2};
  char* p = ans.data();
  std::memcpy(p, s0.data(), n0);
  std::memcpy(p + n0, s1.data(), n1);
  std::memcpy(p + n0 + n1, s2, m);
  ans.setSafe(t, n0 + n1 + n2);
  return ans;
}

inline bool operator<(const il::String& s0, const il::String& s1) {
  const int compare = std::strcmp(s0.data(), s1.data());
  return compare < 0;
}

inline bool operator<(const char* s0, const il::String& s1) {
  const int compare = std::strcmp(s0, s1.data());
  return compare < 0;
}

inline il::String toString(const std::string& s) {
  return il::String{il::StringType::Bytes, s.c_str(),
                    static_cast<il::int_t>(s.size())};
}

inline il::String toString(const char* s) {
  return il::String{il::StringType::Bytes, s, il::size(s)};
}

}  // namespace il

#endif  // IL_STRING_H
