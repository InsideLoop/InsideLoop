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
#include <il/base.h>

#include <il/io/io_base.h>

#include <il/Array.h>
#include <il/String.h>
#include <il/StringView.h>
#include <il/container/string/algorithm_string.h>

namespace il {

typedef il::HashMap<il::String, il::Dynamic> Toml;

class TomlParser {
 private:
  static const il::int_t max_line_length_ = 200;
  char buffer_line_[max_line_length_ + 1];
  il::int_t line_number_;
  std::FILE* file_;

 public:
  TomlParser();
  il::HashMap<il::String, il::Dynamic> parse(const il::String& filename,
                                             il::io_t, il::Status& status);
  il::ConstStringView skip_whitespace_and_comments(il::ConstStringView string,
                                                   il::io_t,
                                                   il::Status& status);
  il::Dynamic parse_value(il::io_t, il::ConstStringView& string,
                          il::Status& status);
  il::Dynamic parse_array(il::io_t, il::ConstStringView& string,
                          il::Status& status);
  il::Dynamic parse_value_array(il::DynamicType value_type, il::io_t,
                                il::ConstStringView& string,
                                il::Status& status);
  il::Dynamic parse_inline_table(il::io_t, il::ConstStringView& string,
                                 il::Status& status);
  void parse_key_value(il::io_t, il::ConstStringView& string,
                       il::HashMap<il::String, il::Dynamic>& toml,
                       il::Status& status);
  void check_end_of_line_or_comment(il::ConstStringView string, il::io_t,
                                    il::Status& status);
  il::String current_line() const;
  il::Dynamic parse_number(il::io_t, il::ConstStringView& string,
                           il::Status& status);
  il::DynamicType parse_type(il::ConstStringView string, il::io_t,
                             il::Status& status);
  il::Dynamic parse_boolean(il::io_t, il::ConstStringView& string,
                            Status& status);
  il::String parse_string_literal(char delimiter, il::io_t,
                                  il::ConstStringView& string,
                                  il::Status& status);
  il::String parse_escape_code(il::io_t, il::ConstStringView& string,
                               il::Status& status);
  il::String parse_key(char end, il::io_t, il::ConstStringView& string,
                       il::Status& status);
  void parse_table(il::io_t, il::ConstStringView& string,
                   il::HashMap<il::String, il::Dynamic>*& toml,
                   il::Status& status);
  void parse_single_table(il::io_t, il::ConstStringView& string,
                          il::HashMap<il::String, il::Dynamic>*& toml,
                          il::Status& status);
  il::Dynamic parse_string(il::io_t, il::ConstStringView& string,
                           il::Status& status);

 private:
  static bool is_digit(char c);
};

template <>
class SaveHelper<il::HashMap<il::String, il::Dynamic>> {
 public:
  static void save(const il::HashMap<il::String, il::Dynamic>& toml,
                   const il::String& filename, il::io_t, il::Status& status) {
    std::FILE* file = std::fopen(filename.c_string(), "wb");
    if (!file) {
      status.set_error(il::ErrorCode::file_not_found);
      return;
    }

    for (il::int_t i = toml.first(); i != toml.last(); i = toml.next(i)) {
      int error0 = std::fputs(toml.key(i).c_string(), file);
      int error1 = std::fputs(" = ", file);
      int error2;
      int error3;
      int error4;
      switch (toml.value(i).type()) {
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
          error3 = std::fputs(toml.value(i).as_const_string().c_string(), file);
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
      status.set_error(il::ErrorCode::cannot_close_file);
      return;
    }

    status.set_ok();
    return;
  }
};

il::HashMap<il::String, il::Dynamic> parse(const il::String& filename, il::io_t,
                                           il::Status& status);

template <>
class LoadHelper<il::HashMap<il::String, il::Dynamic>> {
 public:
  static il::HashMap<il::String, il::Dynamic> load(const il::String& filename,
                                                   il::io_t,
                                                   il::Status& status) {
    il::TomlParser parser{};
    return parser.parse(filename, il::io, status);
  }
};
}

#endif  // IL_TOML_H