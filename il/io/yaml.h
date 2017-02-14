//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_YAML_H
#define IL_YAML_H

#include <il/base.h>
#include <il/io/io_base.h>

#include <il/String.h>
#include <il/StringView.h>
#include <il/Array.h>
#include <il/container/string/algorithm_string.h>

namespace il {

enum class Type {
  empty,
  boolean,
  integer,
  floating_point,
  string
};

class DynamicType {
 private:
  Type type_;
  il::int_t int_;
  bool boolean_;
  double float_;
  il::String string_;

 public:
  DynamicType();
  DynamicType(bool value);
  DynamicType(int n);
  DynamicType(il::int_t n);
  DynamicType(double x);
  DynamicType(const il::String &string);
  DynamicType(const char* string);
  Type type() const;
  bool boolean() const;
  il::int_t integer() const;
  double floating_point() const;
  const il::String &string() const;
};

DynamicType::DynamicType() : string_{} {
  type_ = il::Type::empty;
}

DynamicType::DynamicType(bool value) : string_{} {
  type_ = il::Type::boolean;
  boolean_ = value;
}

DynamicType::DynamicType(int n) : string_{} {
  type_ = il::Type::integer;
  int_ = n;
}

DynamicType::DynamicType(il::int_t n) : string_{} {
  type_ = il::Type::integer;
  int_ = n;
}

DynamicType::DynamicType(double x) : string_{} {
  type_ = il::Type::floating_point;
  float_ = x;
}

DynamicType::DynamicType(const il::String &string) : string_{string} {
  type_ = il::Type::string;
}

DynamicType::DynamicType(const char* string) : string_{string} {
  type_ = il::Type::string;
}

Type DynamicType::type() const {
  return type_;
}

bool DynamicType::boolean() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::boolean);

  return boolean_;
}

il::int_t DynamicType::integer() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::integer);

  return int_;
}

double DynamicType::floating_point() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::floating_point);

  return float_;
}

const il::String &DynamicType::string() const {
  IL_EXPECT_MEDIUM(type_ == il::Type::string);

  return string_;
}

class Yaml {
 private:
  il::Array<il::String> key_;
  il::Array<il::DynamicType> value_;

 public:
  Yaml();
  void append(const char* key, const il::DynamicType& value);
  void append(const il::String& key, const il::DynamicType& value);
  il::int_t search(const il::String &key) const;
  bool found(il::int_t i) const;
  il::Type type(il::int_t i) const;
  const il::String& key(il::int_t) const;
  bool value_boolean(il::int_t i) const;
  il::int_t value_integer(il::int_t i) const;
  double value_floating_point(il::int_t i) const;
  const il::String& value_string(il::int_t i) const;
  il::int_t size() const;
};

Yaml::Yaml() : key_{}, value_{} {}

void Yaml::append(const char* key, const il::DynamicType& value) {
  key_.append(key);
  value_.append(value);
}

void Yaml::append(const il::String& key, const il::DynamicType& value) {
  key_.append(key);
  value_.append(value);
}

il::int_t Yaml::search(const il::String &key) const {
  for (il::int_t i = 0; i < key_.size(); ++i) {
    if (key == key_[i]) {
      return i;
    }
  }
  return -1;
}

bool Yaml::found(il::int_t i) const {
  return i >= 0;
}

il::Type Yaml::type(il::int_t i) const {
  return value_[i].type();
}

const il::String& Yaml::key(il::int_t i) const {
  return key_[i];
}

bool Yaml::value_boolean(il::int_t i) const {
  return value_[i].boolean();
}

il::int_t Yaml::value_integer(il::int_t i) const {
  return value_[i].integer();
}

double Yaml::value_floating_point(il::int_t i) const {
  return value_[i].floating_point();
}

const il::String& Yaml::value_string(il::int_t i) const {
  return value_[i].string();
}


il::int_t Yaml::size() const {
  return key_.size();
}

template <>
class SaveHelper<il::Yaml> {
 public:
  static void save(const il::Yaml& yaml, const il::String& filename, il::io_t,
                   il::Status& status) {
    std::FILE* file = std::fopen(filename.c_string(), "wb");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return;
    }

    for (il::int_t i = 0; i < yaml.size(); ++i) {
      int error0 = std::fputs(yaml.key(i).c_string(), file);
      int error1 = std::fputs(": ", file);
      int error2;
      int error3;
      switch (yaml.type(i)) {
        case il::Type::boolean:
          error2 = std::fputs("!!bool ", file);
          if (yaml.value_boolean(i)){
            error3 = std::fputs("Yes", file);
          } else {
            error3 = std::fputs("No", file);
          }
          break;
        case il::Type::integer:
          error2 = std::fputs("!!int ", file);
          error3 = std::fprintf(file, "%td", yaml.value_integer(i));
          break;
        case il::Type::floating_point:
          error2 = std::fputs("!!float ", file);
          error3 = std::fprintf(file, "%e", yaml.value_floating_point(i));
          break;
        case il::Type::string:
          error2 = std::fputs("!!str ", file);
          error3 = std::fputs(yaml.value_string(i).c_string(), file);
          break;
        default:
          IL_UNREACHABLE;
      }
      error2 = std::fputs("\n", file);
      IL_UNUSED(error0);
      IL_UNUSED(error1);
      IL_UNUSED(error2);
      IL_UNUSED(error3);
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
class LoadHelper<il::Yaml> {
 public:
  static il::Yaml load(const il::String& filename, il::io_t,
                       il::Status& status) {
    il::Yaml yaml{};

    std::FILE* file = std::fopen(filename.c_string(), "r+b");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return yaml;
    }

    const int max_length_line = 200;
    char buffer [max_length_line + 1];
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
      const il::int_t i0 = il::search(": ", line);
      if (i0 == -1) {
        status.set(il::ErrorCode::wrong_file_format);
        return yaml;
      }
      const il::String key = il::String(buffer, i0);

      // Get the type of the value
      if (i0 + 3 >= line.size() || line[i0 + 2] != '!' || line[i0 + 3] != '!') {
        status.set(il::ErrorCode::wrong_file_format);
        return yaml;
      }
      line = line.substring(i0 + 4, line.size());
      const il::int_t i1 = il::search(" ", line);
      if (i1 == -1) {
        status.set(il::ErrorCode::wrong_file_format);
        return yaml;
      }
      const il::String type_string{line.begin(), i1};
      il::Type type;
      if (type_string == "bool") {
        type = il::Type::boolean;
      } else if (type_string == "int") {
        type = il::Type::integer;
      } else if (type_string == "float") {
        type = il::Type::floating_point;
      } else if (type_string == "str") {
        type = il::Type::string;
      } else {
        status.set(il::ErrorCode::wrong_file_format);
        return yaml;
      }

      // Get the value
      line = line.substring(i1 + 1, line.size());
      const il::String value_string{line.begin(), line.size()};
      il::DynamicType value;
      switch (type) {
        case il::Type::boolean:
          if (value_string == "Yes") {
            value = true;
          } else if (value_string == "No") {
            value = false;
          } else {
            status.set(il::ErrorCode::wrong_file_format);
            return yaml;
          }
          break;
        case il::Type::integer:
        {
          il::int_t i = std::atol(value_string.c_string());
          value = i;
        } break;
        case il::Type::floating_point:
        {
          double x = std::atof(value_string.c_string());
          value = x;

        } break;
        case il::Type::string:
        {
          value = value_string;
        } break;
        default:
          IL_UNREACHABLE;
      }

      yaml.append(key, value);
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return yaml;
    }

    status.set(il::ErrorCode::ok);
    return yaml;
  }
};

}

#endif // IL_YAML_H
