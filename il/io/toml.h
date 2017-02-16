//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_TOML_H
#define IL_TOML_H

#include <il/base.h>
#include <il/io/io_base.h>

#include <il/Array.h>
#include <il/String.h>
#include <il/StringView.h>
#include <il/container/string/algorithm_string.h>

namespace il {

enum class Type { empty, boolean, integer, floating_point, string, array };

class DynamicType {
 private:
  Type type_;
  il::int_t integer_;
  bool boolean_;
  double floating_point_;
  il::String string_;

 public:
  DynamicType();
  DynamicType(bool value);
  DynamicType(int n);
  DynamicType(il::int_t n);
  DynamicType(double x);
  DynamicType(const il::String& string);
  DynamicType(const char* string);
  Type type() const;
  bool boolean() const;
  il::int_t integer() const;
  double floating_point() const;
  const il::String& string() const;
};

DynamicType::DynamicType() : string_{} { type_ = il::Type::empty; }

DynamicType::DynamicType(bool value) : string_{} {
  type_ = il::Type::boolean;
  boolean_ = value;
}

DynamicType::DynamicType(int n) : string_{} {
  type_ = il::Type::integer;
  integer_ = n;
}

DynamicType::DynamicType(il::int_t n) : string_{} {
  type_ = il::Type::integer;
  integer_ = n;
}

DynamicType::DynamicType(double x) : string_{} {
  type_ = il::Type::floating_point;
  floating_point_ = x;
}

DynamicType::DynamicType(const il::String& string) : string_{string} {
  type_ = il::Type::string;
}

DynamicType::DynamicType(const char* string) : string_{string} {
  type_ = il::Type::string;
}

Type DynamicType::type() const { return type_; }

bool DynamicType::boolean() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::boolean);

  return boolean_;
}

il::int_t DynamicType::integer() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::integer);

  return integer_;
}

double DynamicType::floating_point() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::floating_point);

  return floating_point_;
}

const il::String& DynamicType::string() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::string);

  return string_;
}

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

inline il::Type type(il::ConstStringView string) {
  if (string[0] == '"' || string[0] == '\'') {
    return il::Type::string;
  } else if (is_digit(string[0]) || string[0] == '-' || string[0] == '+') {
    il::int_t i = 0;
    if (string[0] == '-' || string[0] == '+') {
      ++i;
    }
    while (i < string.size() && is_digit(string[i])) {
      ++i;
    }
    if (i < string.size() && string[i] == '.') {
      ++i;
      while (i < string.size() && is_digit(string[i])) {
        ++i;
      }
      return il::Type::floating_point;
    } else {
      return il::Type::integer;
    }
  } else if (string[0] == 't' || string[0] == 'f') {
    return il::Type::boolean;
  } else {
    il::abort();
  }
  return il::Type::empty;
}

inline il::ConstStringView parse_raw_key(il::ConstStringView string, il::io_t,
                                         il::Status& status) {
  // Remove whitespace
  il::int_t i_begin = 0;
  while (i_begin < string.size() && string[i_begin] == ' ') {
    ++i_begin;
  }
  il::int_t i_end = string.size();
  while (i_end > i_begin && string[i_end - 1] == ' ') {
    --i_end;
  }
  if (i_end == i_begin) {
    status.set(il::ErrorCode::wrong_input);
    status.set_message("Raw key cannot be empty");
    return il::ConstStringView{};
  }

  // Check if the key contains forbidden characters
  il::ConstStringView key = string.substring(i_begin, i_end);
  for (il::int_t i = 0; i < key.size(); ++i) {
    if (key[i] == ' ' || key[i] == '\t') {
      status.set(il::ErrorCode::wrong_input);
      status.set_message("Raw key cannot contain whitespace");
      return il::ConstStringView{};
    }
    if (key[i] == '#') {
      status.set(il::ErrorCode::wrong_input);
      status.set_message("Raw key cannot contain #");
      return il::ConstStringView{};
    }
    if (key[i] == '[' || key[i] == ']') {
      status.set(il::ErrorCode::wrong_input);
      status.set_message("Raw key cannot contain [ or ]");
      return il::ConstStringView{};
    }
  }

  status.set(il::ErrorCode::ok);
  return key;
}

inline il::DynamicType parse_number(il::io_t, il::ConstStringView& string,
                                    il::Status& status) {
  // Find the end of the number
  il::int_t i = 0;
  while (i < string.size() &&
         (is_digit(string[i]) || string[i] == '_' || string[i] == '.' ||
          string[i] == 'e' || string[i] == 'E' || string[i] == '+' ||
          string[i] == '-')) {
    ++i;
  }
  string.shrink_right(i);

  // Skip the +/- sign at the beginning of the string
  i = 0;
  if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
    ++i;
  }

  // Check that there is no leading 0
  if (i + 1 < string.size() && string[i] == '0' && string[i + 1] != '.') {
    status.set(il::ErrorCode::wrong_input);
    status.set_message("Numbers cannot have leading zeros");
    return il::DynamicType{};
  }

  // Skip the numbers
  const il::int_t i_begin_numbers = i;
  while (i < string.size() && is_digit(string[i])) {
    ++i;
    if (i < string.size() && string[i] == '_') {
      ++i;
      if (i == string.size() || !is_digit(string[i + 1])) {
        status.set(il::ErrorCode::wrong_input);
        status.set_message("Malformed number");
        return il::DynamicType{};
      }
    }
  }
  if (i == i_begin_numbers) {
    status.set(il::ErrorCode::wrong_input);
    status.set_message("Malformed number");
    return il::DynamicType{};
  }

  // Detecting if we are a floating point or an integer
  bool is_float;
  if (i < string.size() &&
      (string[i] == '.' || string[i] == 'e' || string[i] == 'E')) {
    is_float = true;
    const bool is_exponent = (string[i] == 'e' || string[i] == 'E');

    ++i;
    if (i == string.size()) {
      status.set(il::ErrorCode::wrong_input);
      status.set_message("Floating point must have trailing digits");
      return il::DynamicType{};
    }

    if (is_exponent) {
      if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
        ++i;
      }
    }
    const il::int_t i_begin_number = i;
    while (i < string.size() && is_digit(string[i])) {
      ++i;
    }
    if (i == i_begin_number) {
      status.set(il::ErrorCode::wrong_input);
      status.set_message("Malformed number");
      return il::DynamicType{};
    }
    if (!is_exponent && i < string.size() &&
        (string[i] == 'e' || string[i] == 'E')) {
      ++i;
      if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
        ++i;
      }
      const il::int_t i_begin_exponent = i;
      while (i < string.size() && is_digit(string[i])) {
        ++i;
      }
      if (i == i_begin_exponent) {
        status.set(il::ErrorCode::wrong_input);
        status.set_message("Malformed number");
        return il::DynamicType{};
      }
    }
  } else {
    is_float = false;
  }

  // Remove the '_' in the number
  il::String float_string{};
  float_string.resize(i);
  il::StringView view{float_string.begin(), i};
  il::int_t k = 0;
  for (il::int_t j = 0; j < i; ++j) {
    if (string[j] != '_') {
      view[k] = string[j];
      ++k;
    }
  }
  float_string.resize(k);

  if (is_float) {
    status.set(il::ErrorCode::ok);
    string.shrink_left(i);
    double x = std::atof(view.begin());
    return il::DynamicType{x};
  } else {
    status.set(il::ErrorCode::ok);
    string.shrink_left(i);
    il::int_t n = std::atoll(view.begin());
    return il::DynamicType{n};
  }
}

inline void parse_array(il::ConstStringView string, il::io_t,
                        il::Status& status) {
  IL_EXPECT_FAST(string.size() >= 1);
  IL_EXPECT_FAST(string[0] == '[');

  // Consume whitespace
  il::int_t i = 1;
  while (i < string.size() && string[i] == ' ') {
    ++i;
  }

  // Special case of an empty array
  if (i >= string.size()) {
    status.set(il::ErrorCode::wrong_input);
    status.set_message("Array is not closed");
    return;
  } else if (string[i] == ']') {
    status.set(il::ErrorCode::ok);
    return;
  }

  il::int_t i_end = i;
  while (
      i_end < string.size() &&
      !(string[i_end] == ',' || string[i_end] == ']' || string[i_end] == '#')) {
    ++i_end;
  }
  il::ConstStringView value_string = string.substring(i, i_end);
  il::Type array_type = type(value_string);
  switch (array_type) {
    case il::Type::boolean:
      break;
    case il::Type::integer:
      break;
    case il::Type::floating_point:
      break;
    case il::Type::string:
      break;
    case il::Type::array:
      break;
    default:
      IL_UNREACHABLE;
  }
}

inline il::DynamicType parse_boolean(il::io_t, il::ConstStringView& string,
                                     Status& status) {
  if (string.size() >= 4 && string[0] == 't' && string[1] == 'r' &&
      string[2] == 'u' && string[3] == 'e') {
    status.set(il::ErrorCode::ok);
    string.shrink_left(4);
    return il::DynamicType{true};
  } else if (string.size() >= 5 && string[0] == 'f' &&
             string[1] == 'a' && string[2] == 'l' &&
             string[3] == 's' && string[4] == 'e') {
    status.set(il::ErrorCode::ok);
    string.shrink_left(5);
    return il::DynamicType{false};
  } else {
    status.set(il::ErrorCode::wrong_input);
    status.set_message("Error when trying to parse a boolean");
    return il::DynamicType{};
  }
}

//inline il::DynamicType parse_value(il::ConstStringView string, il::io_t,
//                                   il::int_t& i, il::Status& status) {
//  // Get the type of the value
//  il::Status parse_status{};
//  il::Type type = parse_type(string, il::io, i, parse_status);
//  if (!parse_status.ok()) {
//    status = parse_status;
//    return il::DynamicType{};
//  }
//
//  il::DynamicType ans{};
//  switch (type) {
//    case il::Type::boolean:
//      ans = parse_boolean(string, il::io, i, parse_status);
//      break;
//    case il::Type::integer:
//    case il::Type::floating_point:
//      ans = parse_number(string, il::io, i, parse_status);
//    case il::Type::string:
//      ans = parse_string(string, il::io, i, parse_status);
//    default:
//      IL_UNREACHABLE;
//  }
//
//  if (!parse_status.ok()) {
//    status = parse_status;
//    return il::DynamicType{};
//  }
//
//  status.set(il::ErrorCode::ok);
//  return ans;
//}

//inline il::Array<il::DynamicType> parse_value_array(il::ConstStringView string,
//                                                    il::Type type, il::io_t,
//                                                    il::int_t& i,
//                                                    il::Status& status) {
//  il::Array<il::DynamicType> ans{};
//
//  while (i < string.size() && string[0] != ']') {
//    // Get the next value and append it to the array
//    il::Status parse_status{};
//    il::DynamicType value = il::parse_value(string, il::io, i, parse_status);
//    if (!parse_status.ok()) {
//      status.set(parse_status.error_code());
//      status.set_message(parse_status.message().c_string());
//      return;
//    }
//    ans.append(value);
//
//    // Check that the array is homogeneous
//    if (value.type() != type) {
//      status.set(il::ErrorCode::wrong_input);
//      status.set_message("Arrays must be homogeneous");
//      return;
//    }
//
//    // Skip whitespaces
//    while (i < string.size() && string[i] == ' ') {
//      ++i;
//    }
//
//    if (string[i] != ',') {
//      break;
//    }
//
//    // Skip whitespaces
//    while (i < string.size() && string[i] == ' ') {
//      ++i;
//    }
//  }
//
//  if (i < string.size()) {
//    ++i;
//  }
//
//  status.set(il::ErrorCode::ok);
//  return ans;
//}

// inline il::ConstStringView skip_whitespace_and_comments(il::ConstStringView
// string) {
//  il::int_t i = 0;
//  while (i < string.size() && string[i] == ' ') {
//    ++i;
//  }
//
//  while (string.size()== 0 || string[i] == '#') {
//    bool
//
//  }
//}

class Toml {
 private:
  il::Array<il::String> key_;
  il::Array<il::DynamicType> value_;

 public:
  Toml();
  void set(const char* key, const il::DynamicType& value);
  void set(const il::String& key, const il::DynamicType& value);
  il::int_t search(const il::String& key) const;
  bool found(il::int_t i) const;
  il::Type type(il::int_t i) const;
  const il::String& key(il::int_t) const;
  bool value_boolean(il::int_t i) const;
  il::int_t value_integer(il::int_t i) const;
  double value_floating_point(il::int_t i) const;
  const il::String& value_string(il::int_t i) const;
  il::int_t size() const;
};

Toml::Toml() : key_{}, value_{} {}

void Toml::set(const char* key, const il::DynamicType& value) {
  key_.append(key);
  value_.append(value);
}

void Toml::set(const il::String& key, const il::DynamicType& value) {
  key_.append(key);
  value_.append(value);
}

il::int_t Toml::search(const il::String& key) const {
  for (il::int_t i = 0; i < key_.size(); ++i) {
    if (key == key_[i]) {
      return i;
    }
  }
  return -1;
}

bool Toml::found(il::int_t i) const { return i >= 0; }

il::Type Toml::type(il::int_t i) const { return value_[i].type(); }

const il::String& Toml::key(il::int_t i) const { return key_[i]; }

bool Toml::value_boolean(il::int_t i) const { return value_[i].boolean(); }

il::int_t Toml::value_integer(il::int_t i) const { return value_[i].integer(); }

double Toml::value_floating_point(il::int_t i) const {
  return value_[i].floating_point();
}

const il::String& Toml::value_string(il::int_t i) const {
  return value_[i].string();
}

il::int_t Toml::size() const { return key_.size(); }

template <>
class SaveHelper<il::Toml> {
 public:
  static void save(const il::Toml& toml, const il::String& filename, il::io_t,
                   il::Status& status) {
    std::FILE* file = std::fopen(filename.c_string(), "wb");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return;
    }

    for (il::int_t i = 0; i < toml.size(); ++i) {
      int error0 = std::fputs(toml.key(i).c_string(), file);
      int error1 = std::fputs(" = ", file);
      int error2;
      int error3;
      int error4;
      switch (toml.type(i)) {
        case il::Type::boolean:
          if (toml.value_boolean(i)) {
            error2 = std::fputs("true", file);
          } else {
            error2 = std::fputs("false", file);
          }
          break;
        case il::Type::integer:
          error3 = std::fprintf(file, "%td", toml.value_integer(i));
          break;
        case il::Type::floating_point:
          error3 = std::fprintf(file, "%e", toml.value_floating_point(i));
          break;
        case il::Type::string:
          error2 = std::fputs("\"", file);
          error3 = std::fputs(toml.value_string(i).c_string(), file);
          error4 = std::fputs("\"", file);
          break;
        default:
          IL_UNREACHABLE;
      }
      int error5 = std::fputs("\n", file);
      IL_UNUSED(error0);
      IL_UNUSED(error1);
      IL_UNUSED(error2);
      IL_UNUSED(error3);
      IL_UNUSED(error4);
      IL_UNUSED(error5);
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return;
    }

    status.set(il::ErrorCode::ok);
    return;
  }
};

template <>
class LoadHelper<il::Toml> {
 public:
  static il::Toml load(const il::String& filename, il::io_t,
                       il::Status& status) {
    il::Toml toml{};

    std::FILE* file = std::fopen(filename.c_string(), "r+b");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return toml;
    }

    const int max_length_line = 200;
    char buffer[max_length_line + 1];
    while (std::fgets(buffer, max_length_line + 1, file) != nullptr) {
      il::int_t i = 0;
      while (i < max_length_line + 1 && buffer[i] != '\0') {
        ++i;
      }
      il::StringView line{buffer, i};
      IL_EXPECT_FAST(i >= 1);
      if (line[i - 1] == '\n') {
        line = line.substring(0, i - 1);
      }
      // Check if it s a comment line
      il::int_t j = 0;
      while (j < line.size() && line[j] == ' ') {
        ++j;
      }
      if (j == line.size()) {
        continue;
      } else if (line[j] == '#') {
        continue;
      }

      // Get the key
      const il::int_t i0 = il::search("=", line);
      if (i0 == -1) {
        status.set(il::ErrorCode::wrong_file_format);
        return toml;
      }
      il::ConstStringView raw_key{buffer, i0};

      il::Status raw_key_status{};
      il::ConstStringView key =
          il::parse_raw_key(raw_key, il::io, raw_key_status);
      if (!raw_key_status.ok()) {
        status.set(il::ErrorCode::wrong_file_format);
        return toml;
      }

      // Get the type of the value
      line = line.substring(i0 + 1, line.size());
      // Remove whitespace
      il::int_t i_begin = 0;
      while (i_begin < line.size() && line[i_begin] == ' ') {
        ++i_begin;
      }
      il::int_t i_end = line.size();
      while (i_end > i_begin && line[i_end - 1] == ' ') {
        --i_end;
      }
      if (i_end == i_begin) {
        status.set(il::ErrorCode::wrong_input);
        return toml;
      }
      line = line.substring(i_begin, i_end);

      const il::Type type = il::type(line);

      // Get the value
      il::ConstStringView view{line.begin(), line.size()};
      il::Status parse_status{};
      il::DynamicType value{};
      switch (type) {
        case il::Type::boolean:
          value = il::parse_boolean(il::io, view, parse_status);
          if (!parse_status.ok()) {
            status = parse_status;
            return toml;
          }
          break;
        case il::Type::integer:
        case il::Type::floating_point:
          value = il::parse_number(il::io, view, parse_status);
          if (!parse_status.ok()) {
            status = parse_status;
            return toml;
          }
          break;
        case il::Type::string:
          value = il::String{line.begin() + 1, line.size() - 2};
          break;
        default:
          IL_UNREACHABLE;
      }

      toml.set(il::String{key.begin(), key.size()}, value);
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return toml;
    }

    status.set(il::ErrorCode::ok);
    return toml;
  }
};
}

#endif  // IL_TOML_H
