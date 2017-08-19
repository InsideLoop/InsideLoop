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
  char* data_;
  char* size_;

 public:
  ConstStringView();
  ConstStringView(const char* data);
  explicit ConstStringView(const char* data, il::int_t n);

  il::int_t size() const;
  bool isEmpty() const;
  bool operator==(const char* string) const;

  bool startsWith(const char* data) const;
  bool startsWith(char c) const;
  bool startsWithNewLine() const;
  bool startsWithDigit() const;

  bool endsWith(const char* data) const;
  bool endsWith(char c) const;
  bool endsWithNewLine() const;
  bool endsWithDigit() const;

  bool contains(il::int_t i, const char* data) const;
  bool contains(il::int_t i, char c) const;
  bool containsDigit(il::int_t i) const;
  bool containsNewLine(il::int_t i) const;

  il::int_t nextChar(il::int_t, char c) const;
  il::int_t nextDigit(il::int_t) const;
  il::int_t nextNewLine(il::int_t) const;

  void removePrefix(il::int_t i1);
  void removeSuffix(il::int_t i0);
  void trimPrefix();
  void trimSuffix();

  ConstStringView substring(il::int_t i0, il::int_t i1) const;
  ConstStringView suffix(il::int_t n) const;
  ConstStringView prefix(il::int_t n) const;

  int rune(il::int_t i) const;
  il::int_t nextRune(il::int_t i) const;

  const char& operator[](il::int_t i) const;
  const char& back(il::int_t i) const;
  const char* asCString() const;
  const char* begin() const;
  const char* end() const;
};

inline ConstStringView::ConstStringView() {
  data_ = nullptr;
  size_ = nullptr;
}

inline ConstStringView::ConstStringView(const char* data) {
  IL_EXPECT_AXIOM("data must be a null terminated string");

  il::int_t size = 0;
  while (data[size] != 0) {
    ++size;
  }
  data_ = const_cast<char*>(data);
  size_ = const_cast<char*>(data) + size;
}

inline ConstStringView::ConstStringView(const char* data, il::int_t n) {
  IL_EXPECT_MEDIUM(n >= 0);

  data_ = const_cast<char*>(data);
  size_ = const_cast<char*>(data) + n;
}

inline il::int_t ConstStringView::size() const { return size_ - data_; }

inline bool ConstStringView::isEmpty() const { return size_ == data_; }

inline const char& ConstStringView::operator[](il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i];
}

inline bool ConstStringView::containsDigit(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i] >= '0' && data_[i] <= '9';
}

inline bool ConstStringView::startsWithDigit() const {
  IL_EXPECT_MEDIUM(size() > 0);

  return data_[0] >= '0' && data_[0] <= '9';
}

inline bool ConstStringView::containsNewLine(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return data_[i] == '\n' ||
         (i + 1 < size() && data_[i] == '\r' && data_[i + 1] == '\n');
}

inline int ConstStringView::rune(il::int_t i) const {
  unsigned int ans = 0;
  const unsigned char* data = reinterpret_cast<const unsigned char*>(begin());
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

inline il::int_t ConstStringView::nextRune(il::int_t i) const {
  do {
    ++i;
  } while (i < size() && ((data_[i] & 0xC0u) == 0x80u));
  return i;
}

inline const char& ConstStringView::back(il::int_t i) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(size()));

  return size_[-(1 + i)];
}

inline bool ConstStringView::startsWithNewLine() const {
  return (size() > 0 && data_[0] == '\n') ||
         (size() > 1 && data_[0] == '\r' && data_[1] == '\n');
}

inline void ConstStringView::removePrefix(il::int_t i1) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i1) <=
                   static_cast<std::size_t>(size()));

  data_ += i1;
}

inline void ConstStringView::removeSuffix(il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <=
                   static_cast<std::size_t>(size()));

  size_ = data_ + i;
}

inline void ConstStringView::trimPrefix() {
  il::int_t i = 0;
  while (i < size() && (data_[i] == ' ' || data_[i] == '\t')) {
    ++i;
  }
  data_ += i;
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

inline ConstStringView ConstStringView::suffix(il::int_t n) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(size()));

  return ConstStringView{data_ + size() - n, size()};
}

inline ConstStringView ConstStringView::prefix(il::int_t n) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(n) <=
                   static_cast<std::size_t>(size()));

  return ConstStringView{data_, n};
}

inline bool ConstStringView::operator==(const char* string) const {
  bool match = true;
  il::int_t k = 0;
  while (match && k < size() && string[k] != '\0') {
    if (data_[k] != string[k]) {
      match = false;
    }
    ++k;
  }
  return match;
}

inline const char* ConstStringView::asCString() const {
  return reinterpret_cast<const char*>(data_);
}

inline const char* ConstStringView::begin() const { return data_; }

inline const char* ConstStringView::end() const { return size_; }

class StringView : public ConstStringView {
 public:
  explicit StringView(char* data, il::int_t n);
  char& operator[](il::int_t i);
  StringView substring(il::int_t i0, il::int_t i1);
  StringView substring(il::int_t i0);
  char* begin();
  char* asCString();
};

inline StringView::StringView(char* data, il::int_t n) {
  IL_EXPECT_MEDIUM(n >= 0);

  data_ = data;
  size_ = data + n;
}

inline char* StringView::begin() { return data_; }

inline char* StringView::asCString() { return reinterpret_cast<char*>(data_); }

inline char& StringView::operator[](il::int_t i) {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(i) <
                   static_cast<std::size_t>(this->size()));

  return this->data_[i];
}

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
