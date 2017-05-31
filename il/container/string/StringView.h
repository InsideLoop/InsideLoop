//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_STRINGVIEW_H
#define IL_STRINGVIEW_H

// <cstring> is needed for memcpy
#include <cstring>

#include <il/base.h>

namespace il {

class ConstStringView {
 protected:
  std::uint8_t* data_;
  std::uint8_t* size_;

 public:
  ConstStringView();
  ConstStringView(const char* data);
  ConstStringView(const std::uint8_t* data);
  explicit ConstStringView(const char* data, il::int_t n);
  explicit ConstStringView(const std::uint8_t* data, il::int_t n);
  char ascii(il::int_t i) const;
  char ascii_back() const;
  //  const std::uint8_t& operator[](il::int_t i) const;
  //  const std::uint8_t& back() const;
  il::int_t size() const;
  void shrink_left(il::int_t i);
  void shrink_right(il::int_t i);
  ConstStringView substring(il::int_t i0) const;
  ConstStringView substring(il::int_t i0, il::int_t i1) const;
  bool is_empty() const;
  bool operator==(const char* string) const;
  const char* c_string() const;
  const std::uint8_t* begin() const;
  const std::uint8_t* end() const;
};

inline ConstStringView::ConstStringView() {
  data_ = nullptr;
  size_ = nullptr;
}

inline ConstStringView::ConstStringView(const std::uint8_t* data) {
  IL_EXPECT_AXIOM("data must be a null terminated string");

  il::int_t size = 0;
  while (data[size] != 0) {
    ++size;
  }
  data_ = const_cast<std::uint8_t*>(data);
  size_ = const_cast<std::uint8_t*>(data) + size;
}

inline ConstStringView::ConstStringView(const char* data)
    : ConstStringView{reinterpret_cast<const std::uint8_t*>(data)} {}

inline ConstStringView::ConstStringView(const std::uint8_t* data, il::int_t n) {
  IL_EXPECT_MEDIUM(n >= 0);

  data_ = const_cast<std::uint8_t*>(data);
  size_ = const_cast<std::uint8_t*>(data) + n;
}

inline ConstStringView::ConstStringView(const char* data, il::int_t n)
    : ConstStringView{reinterpret_cast<const std::uint8_t*>(data), n} {}

inline char ConstStringView::ascii(il::int_t i) const {
  return static_cast<char>(data_[i]);
}

inline char ConstStringView::ascii_back() const {
  return static_cast<char>(size_[-1]);
}
// inline const std::uint8_t& ConstStringView::operator[](il::int_t i) const {
//  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
//                   static_cast<std::size_t>(size()));
//
//  return data_[i];
//}
//
// inline const std::uint8_t& ConstStringView::back() const {
//  IL_EXPECT_MEDIUM(size() >= 1);
//
//  return size_[-1];
//}

inline il::int_t ConstStringView::size() const { return size_ - data_; }

inline bool ConstStringView::is_empty() const { return size_ == data_; }

inline void ConstStringView::shrink_left(il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));

  data_ += i;
}

inline void ConstStringView::shrink_right(il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));

  size_ = data_ + i;
}

inline ConstStringView ConstStringView::substring(il::int_t i0,
                                                  il::int_t i1) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(i0 <= i1);

  return ConstStringView{data_ + i0, i1 - i0};
}

inline ConstStringView ConstStringView::substring(il::int_t i0) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(size()));

  return ConstStringView{data_ + i0, size()};
}

inline bool ConstStringView::operator==(const char* string) const {
  bool match = true;
  il::int_t k = 0;
  while (match && k < size() && string[k] != '\0') {
    if (data_[k] != static_cast<std::uint8_t>(string[k])) {
      match = false;
    }
    ++k;
  }
  return match;
}

inline const char* ConstStringView::c_string() const {
  return reinterpret_cast<const char*>(data_);
}

inline const std::uint8_t* ConstStringView::begin() const { return data_; }

inline const std::uint8_t* ConstStringView::end() const { return size_; }

class StringView : public ConstStringView {
 public:
  explicit StringView(std::uint8_t* data, il::int_t n);
  explicit StringView(char* data, il::int_t n);
  //  std::uint8_t& operator[](il::int_t i);
  StringView substring(il::int_t i0, il::int_t i1);
  StringView substring(il::int_t i0);
  std::uint8_t* begin();
  char* c_string();
};

inline StringView::StringView(std::uint8_t* data, il::int_t n) {
  IL_EXPECT_MEDIUM(n >= 0);

  data_ = data;
  size_ = data + n;
}

inline StringView::StringView(char* data, il::int_t n)
    : StringView{reinterpret_cast<std::uint8_t*>(data), n} {}

inline std::uint8_t* StringView::begin() { return data_; }

inline char* StringView::c_string() { return reinterpret_cast<char*>(data_); }

// inline std::uint8_t& StringView::operator[](il::int_t i) {
//  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
//      static_cast<std::size_t>(this->size()));
//
//  return this->data_[i];
//}

inline StringView StringView::substring(il::int_t i0, il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(this->size()));
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(this->size()));
  IL_EXPECT_MEDIUM(i0 <= i1);

  return StringView{data_ + i0, i1 - i0};
}

inline StringView StringView::substring(il::int_t i0) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i0) <=
                   static_cast<std::size_t>(this->size()));

  return StringView{data_ + i0, this->size()};
}

}  // namespace il

#endif  // IL_STRINGVIEW_H
