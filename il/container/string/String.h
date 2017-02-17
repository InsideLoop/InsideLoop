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

namespace il {

class String {
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
  //
 private:
  struct LargeString {
    char* data;
    std::size_t size;
    std::size_t capacity_;

    il::int_t capacity() const {
      return static_cast<il::int_t>(capacity_ & capacity_extract_mask_);
    }
    void set_capacity(il::int_t r) {
      capacity_ =
          static_cast<std::size_t>(r) |
          (static_cast<std::size_t>(0x80) << ((sizeof(std::size_t) - 1) * 8));
    }
  };
  union {
    unsigned char bytes_[sizeof(LargeString)];
    char small_[sizeof(LargeString)];
    LargeString large_;
  };
  constexpr static il::int_t max_small_size_ =
      static_cast<il::int_t>(sizeof(LargeString) - 1);
  constexpr static unsigned char category_extract_mask_ = 0xC0;
  constexpr static std::size_t capacity_extract_mask_ =
      ~(static_cast<std::size_t>(category_extract_mask_)
        << ((sizeof(std::size_t) - 1) * 8));

 public:
  String();
  String(const char* data);
  String(const char* data, il::int_t n);
  String(il::int_t n);
  String(const String& s);
  String(String&& s);
  String& operator=(const String& s);
  String& operator=(String&& s);
  ~String();
  void reserve(il::int_t r);
  void append(const char*, il::int_t n);
  void append(const char* data);
  void append(char data);
  void append(il::int_t k, char data);
  void append(const String& s);
  void append(const char* data, const String& s);
  void append(const char* data0, const String& s, const char* data1);
  const char& back() const;
  char& back();
  il::int_t size() const;
  bool is_empty() const;
  void resize(il::int_t n);
  il::int_t capacity() const;
  const char* c_string() const;
  char* begin();
  char* end();
  bool operator==(const il::String& other) const;

  bool is_small() const;
  void set_small_size(il::int_t n);
};

inline String::String() {
  small_[0] = '\0';
  set_small_size(0);
}

inline String::String(const char* data) {
  IL_EXPECT_AXIOM("data must be a null terminated string");

  il::int_t size = 0;
  while (data[size] != '\0') {
    ++size;
  }
  if (size <= max_small_size_) {
    std::memcpy(small_, data, static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = new char[size + 1];
    std::memcpy(large_.data, data, static_cast<std::size_t>(size) + 1);
    large_.size = static_cast<std::size_t>(size);
    large_.set_capacity(size);
  }
}

inline String::String(const char* data, il::int_t n) {
  if (n <= max_small_size_) {
    std::memcpy(small_, data, static_cast<std::size_t>(n));
    small_[n] = '\0';
    set_small_size(n);
  } else {
    large_.data = new char[n + 1];
    std::memcpy(large_.data, data, static_cast<std::size_t>(n));
    large_.data[n] = '\0';
    large_.size = static_cast<std::size_t>(n);
    large_.set_capacity(n);
  }
}

inline String::String(il::int_t n) {
  if (n <= max_small_size_) {
    small_[n] = '\0';
    set_small_size(n);
  } else {
    large_.data = new char[n + 1];
    large_.data[n] = '\0';
    large_.size = static_cast<std::size_t>(n);
    large_.set_capacity(n);
  }
}

inline String::String(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(small_, s.c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = new char[size + 1];
    std::memcpy(large_.data, s.c_string(), static_cast<std::size_t>(size) + 1);
    large_.size = static_cast<std::size_t>(size);
    large_.set_capacity(size);
  }
}

inline String::String(String&& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    std::memcpy(small_, s.c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    large_.data = s.large_.data;
    large_.size = s.large_.size;
    large_.capacity_ = s.large_.capacity_;
    s.small_[0] = '\0';
    s.set_small_size(0);
  }
}

inline String& String::operator=(const String& s) {
  const il::int_t size = s.size();
  if (size <= max_small_size_) {
    if (!is_small()) {
      delete[] large_.data;
    }
    std::memcpy(small_, s.c_string(), static_cast<std::size_t>(size) + 1);
    set_small_size(size);
  } else {
    if (size <= capacity()) {
      std::memcpy(large_.data, s.c_string(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
    } else {
      if (!is_small()) {
        delete[] large_.data;
      }
      large_.data = new char[size + 1];
      std::memcpy(large_.data, s.c_string(),
                  static_cast<std::size_t>(size) + 1);
      large_.size = static_cast<std::size_t>(size);
      large_.set_capacity(size);
    }
  }
  return *this;
}

inline String& String::operator=(String&& s) {
  if (this != &s) {
    const il::int_t size = s.size();
    if (size <= max_small_size_) {
      if (!is_small()) {
        delete[] large_.data;
      }
      std::memcpy(small_, s.c_string(), static_cast<std::size_t>(size) + 1);
      set_small_size(size);
    } else {
      large_.data = s.large_.data;
      large_.size = s.large_.size;
      large_.capacity_ = s.large_.capacity_;
      s.small_[0] = '\0';
      s.set_small_size(0);
    }
  }
  return *this;
}

inline String::~String() {
  if (!is_small()) {
    delete[] large_.data;
  }
}

inline void String::reserve(il::int_t r) {
  IL_EXPECT_FAST(r >= 0);

  const bool old_is_small = is_small();
  if (old_is_small && r <= max_small_size_) {
    return;
  }
  const il::int_t old_capacity = capacity();
  if (r <= old_capacity) {
    return;
  }

  const il::int_t old_size = size();
  char* const new_data = new char[r + 1];
  std::memcpy(new_data, c_string(), static_cast<std::size_t>(old_size) + 1);
  if (!old_is_small) {
    delete[] large_.data;
  }
  large_.data = new_data;
  large_.size = old_size;
  large_.set_capacity(r);
}

inline void String::append(const char* data, il::int_t n) {
  IL_EXPECT_FAST(n >= 0);
  IL_EXPECT_AXIOM("data must point to an array of length at least n");

  const il::int_t old_size = size();

  if (is_small()) {
    if (old_size + n <= max_small_size_) {
      std::memcpy(small_ + old_size, data, static_cast<std::size_t>(n));
      small_[old_size + n] = '\0';
      set_small_size(old_size + n);
    } else {
      const il::int_t new_size = old_size + n;
      const il::int_t alt_capacity = 2 * old_size;
      const il::int_t new_capacity =
          new_size > alt_capacity ? new_size : alt_capacity;
      char* new_data = new char[new_capacity + 1];
      std::memcpy(new_data, small_, static_cast<std::size_t>(old_size));
      std::memcpy(new_data + old_size, data, static_cast<std::size_t>(n));
      new_data[old_size + n] = '\0';
      large_.data = new_data;
      large_.size = new_size;
      large_.set_capacity(new_capacity);
    }
  } else {
    if (old_size + n <= capacity()) {
      std::memcpy(large_.data + old_size, data, static_cast<std::size_t>(n));
      large_.data[old_size + n] = '\0';
      large_.size = old_size + n;
    } else {
      const il::int_t new_size = old_size + n;
      const il::int_t alt_capacity = 2 * old_size;
      const il::int_t new_capacity =
          new_size > alt_capacity ? new_size : alt_capacity;
      char* new_data = new char[new_capacity + 1];
      std::memcpy(new_data, large_.data, static_cast<std::size_t>(old_size));
      std::memcpy(new_data + old_size, data, static_cast<std::size_t>(n));
      new_data[old_size + n]= '\0';
      delete[] large_.data;
      large_.data = new_data;
      large_.size = new_size;
      large_.set_capacity(new_capacity);
    }
  }
}

inline void String::append(const char* data) {
  IL_EXPECT_AXIOM("data must be a null terminated string");

  il::int_t size = 0;
  while (data[size] != '\0') {
    ++size;
  }
  append(data, size);
}

inline void String::append(char data) {
  append(&data, 1);
}

inline void String::append(il::int_t k, char data) {
  for (il::int_t i = 0; i < k; ++i) {
    append(data);
  }
}

inline void String::append(const String& s) {
  append(s.c_string(), s.size());
}

inline void String::append(const char* data, const String& s) {
  append(data);
  append(s.c_string());
}

inline void String::append(const char* data0, const String& s,
                           const char* data1) {
  append(data0);
  append(s.c_string());
  append(data1);
}

inline const char& String::back() const {
  if (is_small()) {
    return small_[size() - 1];
  } else {
    return large_.data[size() - 1];
  }
}

inline char& String::back() {
  if (is_small()) {
    return small_[size() - 1];
  } else {
    return large_.data[size() - 1];
  }
}

inline il::int_t String::size() const {
  if (is_small()) {
    return max_small_size_ - static_cast<il::int_t>(small_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

inline bool String::is_empty() const {
  return size() == 0;
}

inline void String::resize(il::int_t n) {
  IL_EXPECT_FAST(n >= 0);

  const il::int_t old_size = size();
  if (is_small()) {
    if (n <= max_small_size_) {
      small_[n] = '\0';
      set_small_size(n);
    } else {
      char* new_data = new char[n + 1];
      std::memcpy(new_data, small_, static_cast<std::size_t>(old_size));
      new_data[n] = '\0';
      large_.data = new_data;
      large_.size = n;
      large_.set_capacity(n);
    }
  } else {
    if (n <= capacity()) {
      large_.data[n] = '\0';
      large_.size = n;
    } else {
      char* new_data = new char[n + 1];
      std::memcpy(new_data, large_.data, static_cast<std::size_t>(old_size));
      new_data[n]= '\0';
      delete[] large_.data;
      large_.data = new_data;
      large_.size = n;
      large_.set_capacity(n);
    }
  }
}

inline il::int_t String::capacity() const {
  if (is_small()) {
    return max_small_size_;
  } else {
    return large_.capacity();
  }
}

inline const char* String::c_string() const {
  if (is_small()) {
    return small_;
  } else {
    return large_.data;
  }
}

inline char* String::begin() {
  if (is_small()) {
    return small_;
  } else {
    return large_.data;
  }
}

inline char* String::end() {
  if (is_small()) {
    return small_ + size();
  } else {
    return large_.data + size();
  }
}

inline bool String::is_small() const {
  return (bytes_[max_small_size_] & category_extract_mask_) == 0;
}

inline void String::set_small_size(il::int_t n) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(max_small_size_));

  small_[max_small_size_] = static_cast<char>(max_small_size_ - n);
}

inline bool String::operator==(const il::String& other) const {
  if (size() != other.size()) {
    return false;
  } else {
    const char* p0 = c_string();
    const char* p1 = other.c_string();
    for (il::int_t i = 0; i < size(); ++i) {
      if (p0[i] != p1[i]) {
        return false;
      }
    }
    return true;
  }
}

}

#endif  // IL_STRING_H
