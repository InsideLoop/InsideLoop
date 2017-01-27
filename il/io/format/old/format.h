//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_FORMAT_H
#define IL_FORMAT_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <il/base.h>

namespace il {

enum class FloatStyle { Percent, Fixed, Exponent };

void write_double(double value, int precision, il::FloatStyle style, il::io_t,
                  std::string& stream) {
  if (std::isnan(value)) {
    stream.append("NaN");
    return;
  } else if (std::isinf(value)) {
    stream.append("Inf");
    return;
  }

  char c_letter;
  switch (style) {
    case FloatStyle::Exponent:
      c_letter = 'e';
      break;
    case FloatStyle::Fixed:
      c_letter = 'f';
      break;
    case FloatStyle::Percent:
      c_letter = 'f';
      break;
    default:
      IL_EXPECT_FAST(false);
  }

  std::string spec{};
  spec.append("%.");
  spec.append(std::to_string(precision));
  spec.append(std::string(1, c_letter));

  if (style == FloatStyle::Percent) {
    value *= 100;
  }

  char buffer[32];
  int length = std::snprintf(buffer, sizeof(buffer), spec.c_str(), value);
  if (style == FloatStyle::Percent) {
    ++length;
  }
  stream.append(buffer);
  if (style == FloatStyle::Percent) {
    stream.append("%");
  }
}

template <typename T>
class FormatProvider {
 public:
  static void format(const T& value, const std::string& style, il::io_t,
                     std::string& stream);
};

template <>
class FormatProvider<double> {
 public:
  static void format(double value, const std::string& style, il::io_t,
                     std::string& stream) {
    FloatStyle s;
    switch (style[0]) {
      case 'P':
        s = FloatStyle::Percent;
        break;
      case 'F':
        s = FloatStyle::Fixed;
        break;
      case 'E':
        s = FloatStyle::Exponent;
        break;
      default:
        IL_EXPECT_FAST(false);
    }

    int precision;
    switch (s) {
      case FloatStyle::Exponent:
        precision = 6;
        break;
      default:
        precision = 2;
    }

    write_double(value, precision, s, il::io, stream);
  }
};

template <>
class FormatProvider<int> {
 public:
  static void format(int value, const std::string& style, il::io_t,
                     std::string& stream) {
    switch (style[0]) {
      case 'Y':
        value ? stream.append("YES") : stream.append("NO");
        break;
      case 'y':
        value ? stream.append("yes") : stream.append("no");
        break;
      default:
        IL_EXPECT_FAST(false);
    }
  }
};

enum class AlignStyle { left, center, right };

template <typename T>
void format_align(const T& value, il::AlignStyle align_style, il::int_t amount,
                  const std::string& style, il::io_t, std::string& stream) {
  if (amount == 0) {
    il::FormatProvider<T>::format(value, style, il::io, stream);
  }

  std::string item{};
  il::FormatProvider<T>::format(value, style, il::io, item);
  if (amount <= static_cast<int>(item.size())) {
    stream.append(std::string(amount, '*'));
    return;
  }

  std::size_t pad_amount = static_cast<std::size_t>(amount) - item.size();
  switch (align_style) {
    case il::AlignStyle::left:
      stream.append(item);
      stream.append(std::string(pad_amount, ' '));
      break;
    case il::AlignStyle::center: {
      std::size_t half_pad_amount = pad_amount / 2;
      stream.append(std::string(pad_amount - half_pad_amount, ' '));
      stream.append(item);
      stream.append(std::string(half_pad_amount, ' '));
      break;
    }
    case il::AlignStyle::right: {
      stream.append(std::string(pad_amount, ' '));
      stream.append(item);
      break;

    }
  }
}

template <typename T>
void format_style(const T& value, const std::string& style, il::io_t, std::string& stream) {
  il::AlignStyle align_style;
  il::int_t amount;
  std::string new_style{};
  if (style[0] == ',') {
    il::int_t pos = 1;
    switch(style[pos]) {
      case '-':
        align_style = il::AlignStyle::left;
        ++pos;
        break;
      case '=':
        align_style = il::AlignStyle::center;
        ++pos;
        break;
      case '+':
        align_style = il::AlignStyle::right;
        ++pos;
        break;
      default:
        align_style = il::AlignStyle::right;
    }
    il::int_t pos_column = pos;
    while (pos_column < static_cast<il::int_t>(style.size()) && style[pos_column] != ':') {
      ++pos_column;
    }
    amount = std::stoi(std::string(style.begin() + pos, style.begin() + pos_column));
    if (pos_column < static_cast<il::int_t>(style.size())) {
      new_style = std::string(style.begin() + pos_column + 1, style.end());
    } else {
      new_style = std::string{};
    }
  } else {
    align_style = il::AlignStyle::right;
    amount = 0;
    new_style = std::string(style.begin() + 1, style.end());
  }

  il::format_align(value, align_style, amount, new_style, il::io, stream);
}

}

#endif  // IL_FORMAT_H
