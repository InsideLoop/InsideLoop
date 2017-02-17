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

bool is_digit(char c);

il::DynamicType parse_type(il::ConstStringView string, il::io_t,
                           il::Status& status);

il::Dynamic parse_boolean(il::io_t, il::ConstStringView& string,
                          Status& status);

il::Dynamic parse_number(il::io_t, il::ConstStringView& string,
                         il::Status& status);

il::String parse_escape_code(il::io_t, il::ConstStringView& string,
                             il::Status& status);

void parse_key_value(il::io_t, il::ConstStringView& string,
                     il::HashMap<il::String, il::Dynamic>& toml,
                     il::Status& status);

il::String parse_key(char end, il::io_t, il::ConstStringView& string,
                     il::Status& status);

il::Dynamic parse_value(il::io_t, il::ConstStringView& string,
                        il::Status& status);

// void parse_table(il::io_t, il::ConstStringView& string, il::Status& status);
//
// void parse_single_table(il::io_t, il::ConstStringView& string, il::Status&
// status);
//
// inline void parse_array(il::ConstStringView string, il::io_t,
//                        il::Status& status) {
//  IL_EXPECT_FAST(string.size() >= 1);
//  IL_EXPECT_FAST(string[0] == '[');
//
//  // Consume whitespace
//  il::int_t i = 1;
//  while (i < string.size() && string[i] == ' ') {
//    ++i;
//  }
//
//  // Special case of an empty array
//  if (i >= string.size()) {
//    status.set_error(il::ErrorCode::wrong_input);
//    status.set_message("Array is not closed");
//    return;
//  } else if (string[i] == ']') {
//    status.set_error(il::ErrorCode::ok);
//    return;
//  }
//
//  il::int_t i_end = i;
//  while (
//      i_end < string.size() &&
//      !(string[i_end] == ',' || string[i_end] == ']' || string[i_end] == '#'))
//      {
//    ++i_end;
//  }
//  il::ConstStringView value_string = string.substring(i, i_end);
//  il::DynamicType array_type = type(value_string);
//  switch (array_type) {
//    case il::DynamicType::boolean:
//      break;
//    case il::DynamicType::integer:
//      break;
//    case il::DynamicType::floating_point:
//      break;
//    case il::DynamicType::string:
//      break;
//    case il::DynamicType::array:
//      break;
//    default:
//      IL_UNREACHABLE;
//  }
//}
//
//
// inline il::Array<il::DynamicType> parse_value_array(il::ConstStringView
// string,
//                                                    il::DynamicType type,
//                                                    il::io_t,
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
//      status.set_error(il::ErrorCode::wrong_input);
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
//  status.set_error(il::ErrorCode::ok);
//  return ans;
//}
//
//
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
          if (toml.value(i).get_boolean()) {
            error2 = std::fputs("true", file);
          } else {
            error2 = std::fputs("false", file);
          }
          break;
        case il::DynamicType::integer:
          error3 = std::fprintf(file, "%td", toml.value(i).get_integer());
          break;
        case il::DynamicType::floating_point:
          error3 = std::fprintf(file, "%e", toml.value(i).get_floating_point());
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

    status.set_error(il::ErrorCode::ok);
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
    return il::parse(filename, il::io, status);
  }
};
}

#endif  // IL_TOML_H
