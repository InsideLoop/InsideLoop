//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/io/toml.h>

namespace il {

TomlParser::TomlParser() {}

il::ConstStringView TomlParser::skip_whitespace_and_comments(
    il::ConstStringView string, il::io_t, il::Status& status) {
  string = il::remove_whitespace_left(string);
  while (string.is_empty() || string[0] == '\n' || string[0] == '#') {
    const char* error = std::fgets(buffer_line_, max_line_length_ + 1, file_);
    if (error == nullptr) {
      status.set(il::ErrorCode::wrong_input, "Unclosed array");
      return string;
    }
    ++line_number_;
    string = il::ConstStringView{buffer_line_};
    string = il::remove_whitespace_left(string);
  }

  status.set_ok();
  return string;
}

bool TomlParser::is_digit(char c) { return c >= '0' && c <= '9'; }

il::DynamicType TomlParser::parse_type(il::ConstStringView string, il::io_t,
                                       il::Status& status) {
  if (string[0] == '"' || string[0] == '\'') {
    status.set_ok();
    return il::DynamicType::string;
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
      status.set_ok();
      return il::DynamicType::floating_point;
    } else {
      status.set_ok();
      return il::DynamicType::integer;
    }
  } else if (string[0] == 't' || string[0] == 'f') {
    status.set_ok();
    return il::DynamicType::boolean;
  } else if (string[0] == '[') {
    status.set_ok();
    return il::DynamicType::array;
  } else if (string[0] == '{') {
    status.set_ok();
    return il::DynamicType::hashmap;
  } else {
    status.set(il::ErrorCode::wrong_input, "Cannot determine type");
    return il::DynamicType::null;
  }
}

il::Dynamic TomlParser::parse_boolean(il::io_t, il::ConstStringView& string,
                                      Status& status) {
  if (string.size() > 3 && string[0] == 't' && string[1] == 'r' &&
      string[2] == 'u' && string[3] == 'e') {
    status.set_error(il::ErrorCode::ok);
    string.shrink_left(4);
    return il::Dynamic{true};
  } else if (string.size() > 4 && string[0] == 'f' && string[1] == 'a' &&
             string[2] == 'l' && string[3] == 's' && string[4] == 'e') {
    status.set_error(il::ErrorCode::ok);
    string.shrink_left(5);
    return il::Dynamic{false};
  } else {
    status.set(il::ErrorCode::wrong_input,
               "Error when trying to parse a boolean");
    return il::Dynamic{};
  }
}

il::Dynamic TomlParser::parse_number(il::io_t, il::ConstStringView& string,
                                     il::Status& status) {
  // Skip the +/- sign at the beginning of the string
  il::int_t i = 0;
  if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
    ++i;
  }

  // Check that there is no leading 0
  if (i + 1 < string.size() && string[i] == '0' && string[i + 1] != '.') {
    status.set_error(il::ErrorCode::wrong_input);
    status.set_message("Numbers cannot have leading zeros");
    return il::Dynamic{};
  }

  // Skip the digits (there should be at least one) before the '.' ot the 'e'
  const il::int_t i_begin_number = i;
  while (i < string.size() && is_digit(string[i])) {
    ++i;
    if (i < string.size() && string[i] == '_') {
      ++i;
      if (i == string.size() || !is_digit(string[i + 1])) {
        status.set_error(il::ErrorCode::wrong_input);
        status.set_message("Malformed number");
        return il::Dynamic{};
      }
    }
  }
  if (i == i_begin_number) {
    status.set_error(il::ErrorCode::wrong_input);
    status.set_message("Malformed number");
    return il::Dynamic{};
  }

  // Detecting if we are a floating point or an integer
  bool is_float;
  if (i < string.size() &&
      (string[i] == '.' || string[i] == 'e' || string[i] == 'E')) {
    is_float = true;

    const bool is_exponent = (string[i] == 'e' || string[i] == 'E');
    ++i;

    if (i == string.size()) {
      status.set_error(il::ErrorCode::wrong_input);
      status.set_message("Malformed floating point");
      return il::Dynamic{};
    }

    // Skip the +/- if we have an exponent
    if (is_exponent) {
      if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
        ++i;
      }
    }

    // Skip the digits (there should be at least one). Note that trailing 0
    // are accepted.
    const il::int_t i_begin_number = i;
    while (i < string.size() && is_digit(string[i])) {
      ++i;
    }
    if (i == i_begin_number) {
      status.set_error(il::ErrorCode::wrong_input);
      status.set_message("Malformed number");
      return il::Dynamic{};
    }

    // If we were after the dot, we might have reached the exponent
    if (!is_exponent && i < string.size() &&
        (string[i] == 'e' || string[i] == 'E')) {
      ++i;

      // Skip the +/- and then the digits (there should be at least one)
      if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
        ++i;
      }
      const il::int_t i_begin_exponent = i;
      while (i < string.size() && is_digit(string[i])) {
        ++i;
      }
      if (i == i_begin_exponent) {
        status.set_error(il::ErrorCode::wrong_input);
        status.set_message("Malformed number");
        return il::Dynamic{};
      }
    }
  } else {
    is_float = false;
  }

  // Remove the '_' in the number
  il::String number{i};
  il::StringView view{number.begin(), i};
  il::int_t k = 0;
  for (il::int_t j = 0; j < i; ++j) {
    if (string[j] != '_') {
      view[k] = string[j];
      ++k;
    }
  }
  number.resize(k);

  if (is_float) {
    status.set_error(il::ErrorCode::ok);
    string.shrink_left(i);
    double x = std::atof(view.begin());
    return il::Dynamic{x};
  } else {
    status.set_error(il::ErrorCode::ok);
    string.shrink_left(i);
    il::int_t n = std::atoll(view.begin());
    return il::Dynamic{n};
  }
}

il::Dynamic TomlParser::parse_string(il::io_t, il::ConstStringView& string,
                                     il::Status& status) {
  IL_EXPECT_FAST(string[0] == '"' || string[0] == '\'');

  const char delimiter = string[0];

  il::Status parse_status{};
  il::Dynamic ans =
      parse_string_literal(delimiter, il::io, string, parse_status);
  if (!parse_status.ok()) {
    status = parse_status;
    return ans;
  }

  status.set_ok();
  return ans;
}

il::String TomlParser::parse_string_literal(char delimiter, il::io_t,
                                            il::ConstStringView& string,
                                            il::Status& status) {
  il::String ans{};

  string.shrink_left(1);
  while (!string.is_empty()) {
    if (delimiter == '"' && string[0] == '\\') {
      il::Status parse_status{};
      ans.append(parse_escape_code(il::io, string, parse_status));
      if (!parse_status.ok()) {
        status = parse_status;
        return ans;
      }
    } else if (string[0] == delimiter) {
      string.shrink_left(1);
      string = il::remove_whitespace_left(string);
      status.set_ok();
      return ans;
    } else {
      ans.append(string[0]);
      string.shrink_left(1);
    }
  }

  status.set(il::ErrorCode::wrong_input, "Unterminated string literal");
  return ans;
}

il::String TomlParser::parse_escape_code(il::io_t, il::ConstStringView& string,
                                         il::Status& status) {
  IL_EXPECT_FAST(string.size() > 0 && string[0] == '\\');

  il::String ans{};
  il::int_t i = 1;
  if (i == string.size()) {
    status.set_error(il::ErrorCode::wrong_input);
    status.set_message("Invalid escape sequence");
    return ans;
  }

  char value;
  switch (string[i]) {
    case 'b':
      value = '\b';
      break;
    case 't':
      value = '\t';
      break;
    case 'n':
      value = '\n';
      break;
    case 'f':
      value = '\f';
      break;
    case 'r':
      value = '\r';
      break;
    case '"':
      value = '"';
      break;
    case '\\':
      value = '\\';
      break;
    case 'u':
    case 'U':
      status.set(il::ErrorCode::wrong_input, "Unicode not handled");
      return ans;
    default:
      status.set(il::ErrorCode::wrong_input, "Invalid escape sequence");
      return ans;
  }

  string.shrink_left(2);
  ans.append(value);
  status.set_ok();
  return ans;
}

il::Dynamic TomlParser::parse_array(il::io_t, il::ConstStringView& string,
                                    il::Status& status) {
  IL_EXPECT_FAST(!string.is_empty() && string[0] == '[');

  il::Dynamic ans{il::DynamicType::array};
  il::Status parse_status{};

  string.shrink_left(1);
  string = skip_whitespace_and_comments(string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = parse_status;
    return ans;
  }

  if (!string.is_empty() && string[0] == ']') {
    string.shrink_left(1);
    status.set_ok();
    return ans;
  }

  il::int_t i = 0;
  while (i < string.size() && string[i] != ',' && string[i] != ']' &&
         string[i] != '#') {
    ++i;
  }
  il::ConstStringView value_string = string.substring(0, i);
  il::DynamicType value_type = parse_type(value_string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = parse_status;
    return ans;
  }

  switch (value_type) {
    case il::DynamicType::null:
    case il::DynamicType::boolean:
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
    case il::DynamicType::string: {
      ans = parse_value_array(value_type, il::io, string, parse_status);
      if (!parse_status.ok()) {
        status = parse_status;
      } else {
        status.set_ok();
      }
      return ans;
    } break;
    default:
      il::abort();
      return ans;
  }
}

il::Dynamic TomlParser::parse_value_array(il::DynamicType value_type, il::io_t,
                                          il::ConstStringView& string,
                                          il::Status& status) {
  il::Dynamic ans{il::DynamicType::array};
  il::Array<il::Dynamic>& array = ans.as_array();
  il::Status parse_status{};

  while (!string.is_empty() && string[0] != ']') {
    il::Dynamic value = parse_value(il::io, string, parse_status);
    if (!parse_status.ok()) {
      status = parse_status;
      return ans;
    }

    if (value.type() == value_type) {
      array.append(value);
    } else {
      status.set(il::ErrorCode::wrong_input, "Array must be heterogeneous");
      return ans;
    }

    string = skip_whitespace_and_comments(string, il::io, parse_status);
    if (!parse_status.ok()) {
      status = parse_status;
      return ans;
    }

    if (string.is_empty() || string[0] != ',') {
      break;
    }

    string.shrink_left(1);
    string = skip_whitespace_and_comments(string, il::io, parse_status);
    if (!parse_status.ok()) {
      status = parse_status;
      return ans;
    }
  }

  if (!string.is_empty()) {
    string.shrink_left(1);
  }
  return ans;
}

il::Dynamic TomlParser::parse_inline_table(il::io_t,
                                           il::ConstStringView& string,
                                           il::Status& status) {
  il::Dynamic ans = il::Dynamic{il::DynamicType::hashmap};
  do {
    string.shrink_left(1);
    if (string.is_empty()) {
      status.set(il::ErrorCode::wrong_input, "Unterminated inline table");
      return ans;
    }
    string = il::remove_whitespace_left(string);
    il::Status parse_status{};
    parse_key_value(il::io, string, ans.as_hashmap(), parse_status);
    if (!parse_status.ok()) {
      status = parse_status;
      return ans;
    }
    string = il::remove_whitespace_left(string);
  } while (string[0] == ',');

  if (string.is_empty() || string[0] != '}') {
    status.set(il::ErrorCode::wrong_input, "Unterminated inline table");
    return ans;
  }

  string.shrink_left(1);
  string = il::remove_whitespace_left(string);

  status.set_ok();
  return ans;
}

void TomlParser::parse_key_value(il::io_t, il::ConstStringView& string,
                                 il::HashMap<il::String, il::Dynamic>& toml,
                                 il::Status& status) {
  il::Status parse_status{};
  il::String key = parse_key('=', il::io, string, parse_status);
  if (!parse_status.ok()) {
    status = parse_status;
    return;
  }

  il::int_t i = toml.search(key);
  if (toml.found(i)) {
    il::String message = "Duplicate key ";
    message.append(key);
    status.set(il::ErrorCode::wrong_input, message.c_string());
    return;
  }

  if (string[0] != '=') {
    il::String message = "No sign '=' after key ";
    message.append(key);
    status.set(il::ErrorCode::wrong_input, message.c_string());
  }
  string.shrink_left(1);
  string = il::remove_whitespace_left(string);

  il::Dynamic value = parse_value(il::io, string, parse_status);
  if (!parse_status.ok()) {
    status = parse_status;
    return;
  }

  toml.set(key, value);
}

il::String TomlParser::parse_key(char end, il::io_t,
                                 il::ConstStringView& string,
                                 il::Status& status) {
  IL_EXPECT_FAST(end == '=' || end == '"' || end == '\'' || end == '@');

  il::String key{};
  string = il::remove_whitespace_left(string);
  IL_EXPECT_FAST(string.size() > 0);

  il::Status parse_status{};
  if (string[0] == '"') {
    //////////////////////////
    // We have a key in a ".."
    //////////////////////////
    string.shrink_left(1);
    while (string.size() > 0) {
      if (string[0] == '\\') {
        il::Status parse_status{};
        key.append(parse_escape_code(il::io, string, parse_status));
        if (!parse_status.ok()) {
          status = parse_status;
          return key;
        }
      } else if (string[0] == '"') {
        string.shrink_left(1);
        string = il::remove_whitespace_left(string);
        status.set_ok();
        return key;
      } else {
        key.append(string[0]);
        string.shrink_left(1);
      }
    }
    status.set(il::ErrorCode::wrong_input, "Unterminated string");
    return key;
  } else {
    /////////////////////////////////////
    // We have a bare key: '..' or ->..<-
    /////////////////////////////////////
    // Search for the end of the key and move back to drop the whitespaces
    if (string[0] == '\'') {
      string.shrink_left(1);
    }
    il::int_t j = 0;
    if (end != '@') {
      while (j < string.size() && string[j] != end) {
        ++j;
      }
    } else {
      while (j < string.size() && string[j] != '.' && string[j] != ']') {
        ++j;
      }
    }
    const il::int_t j_end = j;
    while (j > 0 && (string[j - 1] == ' ' || string[j - 1] == '\t')) {
      --j;
    }
    if (j == 0) {
      status.set(il::ErrorCode::wrong_input, "Raw key cannot be empty");
      return key;
    }

    // Check if the key contains forbidden characters
    il::ConstStringView key_string = string.substring(0, j);
    string.shrink_left(j_end);

    for (il::int_t i = 0; i < key_string.size(); ++i) {
      if (key_string[i] == ' ' || key_string[i] == '\t') {
        status.set(il::ErrorCode::wrong_input,
                   "Raw key cannot contain whitespace");
        return key;
      }
      if (key_string[i] == '#') {
        status.set(il::ErrorCode::wrong_input, "Raw key cannot contain #");
        return key;
      }
      if (key_string[i] == '[' || key_string[i] == ']') {
        status.set(il::ErrorCode::wrong_input, "Raw key cannot contain [ or ]");
        return key;
      }
    }

    key = il::String{key_string.begin(), j};
    status.set_ok();
    return key;
  }
}

il::Dynamic TomlParser::parse_value(il::io_t, il::ConstStringView& string,
                                    il::Status& status) {
  il::Dynamic ans{};

  // Get the type of the value
  il::Status parse_status{};
  il::DynamicType type = parse_type(string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = parse_status;
    return ans;
  }

  switch (type) {
    case il::DynamicType::boolean:
      ans = parse_boolean(il::io, string, parse_status);
      break;
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
      ans = parse_number(il::io, string, parse_status);
      break;
    case il::DynamicType::string:
      ans = parse_string(il::io, string, parse_status);
      break;
    case il::DynamicType::array:
      ans = parse_array(il::io, string, parse_status);
      break;
    case il::DynamicType::hashmap:
      ans = parse_inline_table(il::io, string, parse_status);
      break;
    default:
      IL_UNREACHABLE;
  }

  if (!parse_status.ok()) {
    status = parse_status;
    return ans;
  }

  status.set_error(il::ErrorCode::ok);
  return ans;
}

void TomlParser::parse_table(il::io_t, il::ConstStringView& string,
                             il::HashMap<il::String, il::Dynamic>*& toml,
                             il::Status& status) {
  // Skip the '[' at the beginning of the table
  string.shrink_left(1);

  if (string.is_empty()) {
    status.set(il::ErrorCode::wrong_input, "Unexpected end of table");
    return;
  } else if (string[0] == '[') {
    // Parsing a table array. Not implemented yet
    il::abort();
  } else {
    parse_single_table(il::io, string, toml, status);
    return;
  }
}

void TomlParser::parse_single_table(il::io_t, il::ConstStringView& string,
                                    il::HashMap<il::String, il::Dynamic>*& toml,
                                    il::Status& status) {
  if (string.is_empty() || string[0] == ']') {
    status.set(il::ErrorCode::wrong_input, "Table cannot be empty");
  }

  il::String full_table_name{};
  bool inserted = false;
  while (!string.is_empty() && string[0] != ']') {
    il::Status parse_status{};
    il::String table_name = parse_key('@', il::io, string, parse_status);
    if (!parse_status.ok()) {
      status = parse_status;
      return;
    }
    if (table_name.is_empty()) {
      status.set(il::ErrorCode::wrong_input, "The name cannot be empty");
      return;
    }
    if (!full_table_name.is_empty()) {
      full_table_name.append('.');
    }
    full_table_name.append(table_name);

    il::int_t i = toml->search(table_name);
    if (toml->found(i)) {
      if (toml->value(i).is_hashmap()) {
        toml = &(toml->value(i).as_hashmap());
      } else if (toml->value(i).is_array()) {
        il::abort();
      } else {
        status.set(il::ErrorCode::wrong_input,
                   "The key already exists as a value");
        return;
      }
    } else {
      inserted = true;
      toml->insert(table_name, il::Dynamic{il::DynamicType::hashmap}, il::io,
                   i);
      toml = &(toml->value(i).as_hashmap());
    }

    string = il::remove_whitespace_left(string);
    while (!string.is_empty() && string[0] == '.') {
      string.shrink_left(1);
    }
    string = il::remove_whitespace_left(string);
  }

  // TODO: One should check the redefinition of a table (line 1680)

  string.shrink_left(1);
  string = il::remove_whitespace_left(string);
  if (!string.is_empty() && string[0] != '\n' && string[0] != '#') {
    il::abort();
  }
}

il::HashMap<il::String, il::Dynamic> TomlParser::parse(
    const il::String& filename, il::io_t, il::Status& status) {
  il::HashMap<il::String, il::Dynamic> root_toml{};
  il::HashMap<il::String, il::Dynamic>* pointer_toml = &root_toml;

  file_ = std::fopen(filename.c_string(), "r+b");
  if (!file_) {
    status.set_error(il::ErrorCode::file_not_found);
    return root_toml;
  }
  line_number_ = 1;

  while (std::fgets(buffer_line_, max_line_length_ + 1, file_) != nullptr) {
    il::ConstStringView line{buffer_line_};
    line = il::remove_whitespace_left(line);

    if (line.is_empty() || line[0] == '\n' || line[0] == '#') {
      continue;
    } else if (line[0] == '[') {
      pointer_toml = &root_toml;
      il::Status parse_status{};
      parse_table(il::io, line, pointer_toml, parse_status);
      if (!parse_status.ok()) {
        status = parse_status;
        std::fclose(file_);
        return root_toml;
      }
    } else {
      il::Status parse_status{};
      parse_key_value(il::io, line, *pointer_toml, parse_status);
      if (!parse_status.ok()) {
        status = parse_status;
        std::fclose(file_);
        return root_toml;
      }

      line = il::remove_whitespace_left(line);

      if (!line.is_empty() && line[0] != '\n' && line[0] != '#') {
        il::abort();
      }
    }
  }

  const int error = std::fclose(file_);
  if (error != 0) {
    status.set_error(il::ErrorCode::cannot_close_file);
    return root_toml;
  }

  status.set_ok();
  return root_toml;
}
}
