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

#include <il/Dynamic.h>
#include <il/Status.h>
#include <il/base.h>

#include <il/io/io_base.h>

#include <il/Array.h>
#include <il/String.h>
#include <il/StringView.h>
#include <il/container/string/algorithm_string.h>

namespace il {

typedef il::HashMapArray<il::String, il::Dynamic> Toml;

class TomlParser {
 private:
  static const il::int_t max_line_length_ = 200;
  char buffer_line_[max_line_length_ + 1];
  il::int_t line_number_;
  std::FILE *file_;

 public:
  TomlParser();
  il::HashMapArray<il::String, il::Dynamic> parse(const il::String &filename,
                                                  il::io_t, il::Status &status);
  il::ConstStringView skip_whitespace_and_comments(il::ConstStringView string,
                                                   il::io_t,
                                                   il::Status &status);
  il::Dynamic parse_value(il::io_t, il::ConstStringView &string,
                          il::Status &status);
  il::Dynamic parse_array(il::io_t, il::ConstStringView &string,
                          il::Status &status);
  il::Dynamic parse_value_array(il::DynamicType value_type, il::io_t,
                                il::ConstStringView &string,
                                il::Status &status);
  il::Dynamic parse_object_array(il::DynamicType object_type, char delimiter,
                                 il::io_t, il::ConstStringView &string,
                                 il::Status &status);
  il::Dynamic parse_inline_table(il::io_t, il::ConstStringView &string,
                                 il::Status &status);
  void parse_key_value(il::io_t, il::ConstStringView &string,
                       il::HashMapArray<il::String, il::Dynamic> &toml,
                       il::Status &status);
  void check_end_of_line_or_comment(il::ConstStringView string, il::io_t,
                                    il::Status &status);
  il::String current_line() const;
  il::Dynamic parse_number(il::io_t, il::ConstStringView &string,
                           il::Status &status);
  il::DynamicType parse_type(il::ConstStringView string, il::io_t,
                             il::Status &status);
  il::Dynamic parse_boolean(il::io_t, il::ConstStringView &string,
                            Status &status);
  il::String parse_string_literal(char delimiter, il::io_t,
                                  il::ConstStringView &string,
                                  il::Status &status);
  il::String parse_escape_code(il::io_t, il::ConstStringView &string,
                               il::Status &status);
  il::String parse_key(char end, il::io_t, il::ConstStringView &string,
                       il::Status &status);
  void parse_table(il::io_t, il::ConstStringView &string,
                   il::HashMapArray<il::String, il::Dynamic> *&toml,
                   il::Status &status);
  void parse_single_table(il::io_t, il::ConstStringView &string,
                          il::HashMapArray<il::String, il::Dynamic> *&toml,
                          il::Status &status);
  void parse_table_array(il::io_t, il::ConstStringView &string,
                         il::HashMapArray<il::String, il::Dynamic> *&toml,
                         il::Status &status);
  il::Dynamic parse_string(il::io_t, il::ConstStringView &string,
                           il::Status &status);

 private:
  static bool is_digit(char c);
};

inline void save_array(const il::Array<il::Dynamic> &array, il::io_t,
                       std::FILE *file, il::Status &status) {
  const int error0 = std::fputs("[ ", file);
  IL_UNUSED(error0);
  for (il::int_t j = 0; j < array.size(); ++j) {
    if (j > 0) {
      const int error1 = std::fputs(", ", file);
      IL_UNUSED(error1);
    }
    switch (array[j].type()) {
      case il::DynamicType::boolean: {
        if (array[j].to_boolean()) {
          const int error2 = std::fputs("true", file);
          IL_UNUSED(error2);
        } else {
          const int error2 = std::fputs("false", file);
          IL_UNUSED(error2);
        }
      } break;
      case il::DynamicType::integer: {
        const int error2 = std::fprintf(file, "%td", array[j].to_integer());
        IL_UNUSED(error2);
      } break;
      case il::DynamicType::floating_point: {
        const int error2 =
            std::fprintf(file, "%e", array[j].to_floating_point());
        IL_UNUSED(error2);
      } break;
      case il::DynamicType::string: {
        const int error2 = std::fputs("\"", file);
        const int error3 =
            std::fputs(array[j].as_const_string().as_c_string(), file);
        const int error4 = std::fputs("\"", file);
        IL_UNUSED(error2);
        IL_UNUSED(error3);
        IL_UNUSED(error4);
      } break;
      case il::DynamicType::array: {
        save_array(array[j].as_const_array(), il::io, file, status);
        if (status.is_error()) {
          status.rearm();
          return;
        }
      } break;
      default:
        IL_UNREACHABLE;
    }
  }
  const int error5 = std::fputs(" ]", file);
  IL_UNUSED(error5);
  status.set_ok();
}

inline void save_aux(const il::HashMapArray<il::String, il::Dynamic> &toml,
                     const il::String &name, il::io_t, std::FILE *file,
                     il::Status &status) {
  for (il::int_t i = 0; i != toml.size(); ++i) {
    il::DynamicType type = toml.value(i).type();
    if (type != il::DynamicType::hashmaparray) {
      int error0 = std::fputs(toml.key(i).as_c_string(), file);
      int error1 = std::fputs(" = ", file);
      int error2;
      int error3;
      int error4;
      switch (type) {
        case il::DynamicType::boolean:
          if (toml.value(i).to_boolean()) {
            error2 = std::fputs("true", file);
          } else {
            error2 = std::fputs("false", file);
          }
          break;
        case il::DynamicType::integer:
          error3 = std::fprintf(file, "%td", toml.value(i).to_integer());
          break;
        case il::DynamicType::floating_point:
          error3 = std::fprintf(file, "%e", toml.value(i).to_floating_point());
          break;
        case il::DynamicType::string:
          error2 = std::fputs("\"", file);
          error3 =
              std::fputs(toml.value(i).as_const_string().as_c_string(), file);
          error4 = std::fputs("\"", file);
          break;
        case il::DynamicType::array: {
        } break;
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
    } else {
      const int error0 = std::fputs("\n[", file);
      IL_UNUSED(error0);
      if (name.size() != 0) {
        const int error1 = std::fputs(name.as_c_string(), file);
        const int error2 = std::fputs(".", file);
        IL_UNUSED(error1);
        IL_UNUSED(error2);
      }
      const int error3 = std::fputs(toml.key(i).as_c_string(), file);
      const int error4 = std::fputs("]\n", file);
      save_aux(toml.value(i).as_const_hashmaparray(), toml.key(i), il::io, file,
               status);
      IL_UNUSED(error3);
      IL_UNUSED(error4);
      if (status.is_error()) {
        status.rearm();
        return;
      }
    }
  }

  status.set_ok();
  return;
}

template <>
class SaveHelper<il::HashMapArray<il::String, il::Dynamic>> {
 public:
  static void save(const il::HashMapArray<il::String, il::Dynamic> &toml,
                   const il::String &filename, il::io_t, il::Status &status) {
    std::FILE *file = std::fopen(filename.as_c_string(), "wb");
    if (!file) {
      status.set_error(il::Error::filesystem_file_not_found);
      return;
    }

    il::String root_name{};
    save_aux(toml, root_name, il::io, file, status);
    if (status.is_error()) {
      status.rearm();
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set_error(il::Error::filesystem_cannot_close_file);
      return;
    }

    status.set_ok();
    return;
  }
};

il::HashMapArray<il::String, il::Dynamic> parse(const il::String &filename,
                                                il::io_t, il::Status &status);

template <>
class LoadHelper<il::HashMapArray<il::String, il::Dynamic>> {
 public:
  static il::HashMapArray<il::String, il::Dynamic> load(
      const il::String &filename, il::io_t, il::Status &status) {
    il::TomlParser parser{};
    return parser.parse(filename, il::io, status);
  }
};
}  // namespace il

#endif  // IL_TOML_H