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
  il::int_t size() const;
  il::int_t length() const;
  il::int_t next_cp(il::int_t i) const;
  bool is_char(il::int_t i) const;
  bool is_char(il::int_t i, char c) const;
  bool is_digit(il::int_t i) const;
  bool is_char_back() const;
  bool is_char_back(char c) const;
  std::uint8_t to_cu(il::int_t i) const;
  std::int32_t to_cp(il::int_t i) const;
  char to_char_checked(il::int_t i) const;
  //  char to_char_back() const;
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

inline bool ConstStringView::is_char(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i] < 128;
}

inline bool ConstStringView::is_char(il::int_t i, char c) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(data_[i] < 128);
  IL_EXPECT_MEDIUM(static_cast<std::uint8_t>(c) < 128);

  return data_[i] == c;
}

inline bool ConstStringView::is_digit(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i] >= static_cast<std::uint8_t>('0') &&
         data_[i] <= static_cast<std::uint8_t>('9');
}

inline bool ConstStringView::is_char_back() const {
  IL_EXPECT_MEDIUM(size() > 0);

  return size_[-1] < 128;
}

inline bool ConstStringView::is_char_back(char c) const {
  IL_EXPECT_MEDIUM(size() > 0);
  IL_EXPECT_MEDIUM(static_cast<std::uint8_t>(c) < 128);

  return size_[-1] == c;
}

inline std::uint8_t ConstStringView::to_cu(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i];
}

inline std::int32_t ConstStringView::to_cp(il::int_t i) const {
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

inline char ConstStringView::to_char_checked(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));
  IL_EXPECT_MEDIUM(data_[i] < 128);

  return static_cast<char>(data_[i]);
}

// inline char ConstStringView::to_char_back() const {
//  return static_cast<char>(size_[-1]);
//}

// inline const std::uint8_t& ConstStringView::operator[](il::int_t i) const {
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

inline il::int_t ConstStringView::length() const {
  il::int_t k = 0;
  for (il::int_t i = 0; i < size(); i = next_cp(i)) {
    ++k;
  }
  return k;
}

inline il::int_t ConstStringView::next_cp(il::int_t i) const {
  do {
    ++i;
  } while (i < size() && ((data_[i] & 0xC0u) == 0x80u));
  return i;
}

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
