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

#include <il/base.h>

namespace il {

class String {
 private:
  struct LargeString {
   public:
    char* data;
    std::size_t size;

   private:
    std::size_t capacity_;

   public:
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
  ~String();
  void append(const char* data);
  il::int_t size() const;
  il::int_t capacity() const;
  const char* c_str() const;

  bool is_small() const;
  void set_small_size(il::int_t n);
};

String::String() {
  small_[0] = '\0';
  set_small_size(0);
}

String::String(const char* data) {
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

String::~String() {
  if (!is_small()) {
    delete[] large_.data;
  }
}

void String::append(const char* data) {
  const il::int_t old_size = size();
  il::int_t size = 0;
  while (data[size] != '\0') {
    ++size;
  }
  if (is_small()) {
    if (old_size + size <= max_small_size_) {
      std::memcpy(small_ + old_size, data, static_cast<std::size_t>(size) + 1);
      set_small_size(old_size + size);
    } else {
      char* new_data = new char[old_size + size + 1];
      std::memcpy(new_data, small_, static_cast<std::size_t>(old_size));
      std::memcpy(new_data + old_size, data,
                  static_cast<std::size_t>(size) + 1);
      large_.data = new_data;
      large_.size = old_size + size;
      large_.set_capacity(old_size + size);
    }
  } else {
    if (old_size + size <= capacity()) {
      std::memcpy(large_.data + old_size, data,
                  static_cast<std::size_t>(size) + 1);
      large_.size = old_size + size;
    } else {
      char* new_data = new char[old_size + size + 1];
      std::memcpy(new_data, large_.data, static_cast<std::size_t>(old_size));
      std::memcpy(new_data + old_size, data,
                  static_cast<std::size_t>(size) + 1);
      delete[] large_.data;
      large_.data = new_data;
      large_.size = old_size + size;
      large_.set_capacity(old_size + size);
    }
  }
}

il::int_t String::size() const {
  if (is_small()) {
    return max_small_size_ - static_cast<il::int_t>(small_[max_small_size_]);
  } else {
    return static_cast<il::int_t>(large_.size);
  }
}

il::int_t String::capacity() const {
  if (is_small()) {
    return max_small_size_;
  } else {
    return large_.capacity();
  }
}

const char* String::c_str() const {
  if (is_small()) {
    return small_;
  } else {
    return large_.data;
  }
}

bool String::is_small() const {
  return (bytes_[max_small_size_] & category_extract_mask_) == 0;
}

void String::set_small_size(il::int_t n) {
  IL_ASSERT_PRECOND(static_cast<il::uint_t>(n) <=
                    static_cast<il::uint_t>(max_small_size_));

  small_[max_small_size_] = static_cast<char>(max_small_size_ - n);
}
}

#endif  // IL_STRING_H
