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
  while (string.is_empty() || string.is_ascii(0, '\n') ||
         string.is_ascii(0, '#')) {
    const char* error = std::fgets(buffer_line_, max_line_length_ + 1, file_);
    if (error == nullptr) {
      status.set_error(il::Error::parse_unclosed_array);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
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

void TomlParser::check_end_of_line_or_comment(il::ConstStringView string,
                                              il::io_t, il::Status& status) {
  if (!string.is_empty() && (!string.is_newline(0)) &&
      (!string.is_ascii(0, '#'))) {
    status.set_error(il::Error::parse_unidentified_trailing_character);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
  } else {
    status.set_ok();
  }
}

il::String TomlParser::current_line() const {
  char line[24];
  line[23] = '\0';
  std::sprintf(line, "%td", line_number_);
  il::String ans = line;
  return line;
}

il::Type TomlParser::parse_type(il::ConstStringView string, il::io_t,
                                il::Status& status) {
  if (string.is_ascii(0, '"') || string.is_ascii(0, '\'')) {
    status.set_ok();
    return il::Type::string_t;
  } else if (string.is_digit(0) || string.is_ascii(0, '-') ||
             string.is_ascii(0, '+')) {
    il::int_t i = 0;
    if (string.is_ascii(0, '-') || string.is_ascii(0, '+')) {
      ++i;
    }
    while (i < string.size() && string.is_digit(i)) {
      ++i;
    }
    if (i < string.size() && string.is_ascii(i, '.')) {
      ++i;
      while (i < string.size() && string.is_digit(i)) {
        ++i;
      }
      status.set_ok();
      return il::Type::double_t;
    } else {
      status.set_ok();
      return il::Type::integer_t;
    }
  } else if (string.is_ascii(0, 't') || string.is_ascii(0, 'f')) {
    status.set_ok();
    return il::Type::bool_t;
  } else if (string.is_ascii(0, '[')) {
    status.set_ok();
    return il::Type::array_t;
  } else if (string.is_ascii(0, '{')) {
    status.set_ok();
    return il::Type::hash_map_array_t;
  } else {
    status.set_error(il::Error::parse_cannot_determine_type);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return il::Type::null_t;
  }
}

il::Dynamic TomlParser::parse_boolean(il::io_t, il::ConstStringView& string,
                                      Status& status) {
  if (string.size() > 3 && string.is_ascii(0, 't') && string.is_ascii(1, 'r') &&
      string.is_ascii(2, 'u') && string.is_ascii(3, 'e')) {
    status.set_ok();
    string.remove_prefix(4);
    return il::Dynamic{true};
  } else if (string.size() > 4 && string.is_ascii(0, 'f') &&
             string.is_ascii(1, 'a') && string.is_ascii(2, 'l') &&
             string.is_ascii(3, 's') && string.is_ascii(4, 'e')) {
    status.set_ok();
    string.remove_prefix(5);
    return il::Dynamic{false};
  } else {
    status.set_error(il::Error::parse_bool);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return il::Dynamic{};
  }
}

il::Dynamic TomlParser::parse_number(il::io_t, il::ConstStringView& string,
                                     il::Status& status) {
  // Skip the +/- sign at the beginning of the string
  il::int_t i = 0;
  if (i < string.size() &&
      (string.is_ascii(i, '+') || string.is_ascii(i, '-'))) {
    ++i;
  }

  // Check that there is no leading 0
  if (i + 1 < string.size() && string.is_ascii(i, '0') &&
      !(string.is_ascii(i + 1, '.') || string.is_ascii(i + 1, ' ') ||
        string.is_newline(i + 1))) {
    status.set_error(il::Error::parse_number);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return il::Dynamic{};
  }

  // Skip the digits (there should be at least one) before the '.' ot the 'e'
  const il::int_t i_begin_number = i;
  while (i < string.size() && string.is_digit(i)) {
    ++i;
    if (i < string.size() && string.is_ascii(i, '_')) {
      ++i;
      if (i == string.size() || (!string.is_digit(i + 1))) {
        status.set_error(il::Error::parse_number);
        IL_SET_SOURCE(status);
        status.set_info("line", line_number_);
        return il::Dynamic{};
      }
    }
  }
  if (i == i_begin_number) {
    status.set_error(il::Error::parse_number);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return il::Dynamic{};
  }

  // Detecting if we are a floating point or an integer
  bool is_float;
  if (i < string.size() &&
      (string.is_ascii(i, '.') || string.is_ascii(i, 'e') ||
       string.is_ascii(i, 'E'))) {
    is_float = true;

    const bool is_exponent =
        (string.is_ascii(i, 'e') || string.is_ascii(i, 'E'));
    ++i;

    if (i == string.size()) {
      status.set_error(il::Error::parse_double);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return il::Dynamic{};
    }

    // Skip the +/- if we have an exponent
    if (is_exponent) {
      if (i < string.size() &&
          (string.is_ascii(i, '+') || string.is_ascii(i, '-'))) {
        ++i;
      }
    }

    // Skip the digits (there should be at least one). Note that trailing 0
    // are accepted.
    const il::int_t i_begin_number = i;
    while (i < string.size() && string.is_digit(i)) {
      ++i;
    }
    if (i == i_begin_number) {
      status.set_error(il::Error::parse_double);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return il::Dynamic{};
    }

    // If we were after the dot, we might have reached the exponent
    if (!is_exponent && i < string.size() &&
        (string.is_ascii(i, 'e') || string.is_ascii(i, 'E'))) {
      ++i;

      // Skip the +/- and then the digits (there should be at least one)
      if (i < string.size() &&
          (string.is_ascii(i, '+') || string.is_ascii(i, '-'))) {
        ++i;
      }
      const il::int_t i_begin_exponent = i;
      while (i < string.size() && string.is_digit(i)) {
        ++i;
      }
      if (i == i_begin_exponent) {
        status.set_error(il::Error::parse_double);
        IL_SET_SOURCE(status);
        status.set_info("line", line_number_);
        return il::Dynamic{};
      }
    }
  } else {
    is_float = false;
  }

  // Remove the '_' in the number
  il::String number{};
  number.reserve(i);
  for (il::int_t j = 0; j < i; ++j) {
    if (!string.is_ascii(j, '_')) {
      number.append(string.to_ascii(j));
    }
  }
  il::ConstStringView view{number.as_c_string(), i};

  if (is_float) {
    status.set_ok();
    string.remove_prefix(i);
    double x = std::atof(view.as_c_string());
    return il::Dynamic{x};
  } else {
    status.set_ok();
    string.remove_prefix(i);
    il::int_t n = std::atoll(view.as_c_string());
    return il::Dynamic{n};
  }
}

il::Dynamic TomlParser::parse_string(il::io_t, il::ConstStringView& string,
                                     il::Status& status) {
  IL_EXPECT_FAST(string.is_ascii(0, '"') || string.is_ascii(0, '\''));

  const char delimiter = string.to_ascii(0);

  il::Status parse_status{};
  il::Dynamic ans =
      parse_string_literal(delimiter, il::io, string, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return ans;
  }

  status.set_ok();
  return ans;
}

il::String TomlParser::parse_string_literal(char delimiter, il::io_t,
                                            il::ConstStringView& string,
                                            il::Status& status) {
  il::String ans{};

  string.remove_prefix(1);
  while (!string.is_empty()) {
    if (delimiter == '"' && string.is_ascii(0, '\\')) {
      il::Status parse_status{};
      ans.append(parse_escape_code(il::io, string, parse_status));
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        return ans;
      }
    } else if (string.is_ascii(0, delimiter)) {
      string.remove_prefix(1);
      string = il::remove_whitespace_left(string);
      status.set_ok();
      return ans;
    } else {
      // TODO: I am not sure what will happen with a Unicode string
      ans.append(string.to_code_point(0));
      string.remove_prefix(string.next_code_point(0));
    }
  }

  status.set_error(il::Error::parse_string);
  IL_SET_SOURCE(status);
  status.set_info("line", line_number_);
  return ans;
}

il::String TomlParser::parse_escape_code(il::io_t, il::ConstStringView& string,
                                         il::Status& status) {
  IL_EXPECT_FAST(string.size() > 0 && string.is_ascii(0, '\\'));

  il::String ans{};
  il::int_t i = 1;
  if (i == string.size()) {
    status.set_error(il::Error::parse_string);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return ans;
  }

  char value;
  switch (string.to_ascii(i)) {
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
    case 'U': {
      status.set_error(il::Error::parse_string);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return ans;
    } break;
    default:
      status.set_error(il::Error::parse_string);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return ans;
  }

  string.remove_prefix(2);
  ans.append(value);
  status.set_ok();
  return ans;
}

il::Dynamic TomlParser::parse_array(il::io_t, il::ConstStringView& string,
                                    il::Status& status) {
  IL_EXPECT_FAST(!string.is_empty() && string.is_ascii(0, '['));

  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Status parse_status{};

  string.remove_prefix(1);
  string = skip_whitespace_and_comments(string, il::io, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return ans;
  }

  if (!string.is_empty() && string.is_ascii(0, ']')) {
    string.remove_prefix(1);
    status.set_ok();
    return ans;
  }

  il::int_t i = 0;
  while (i < string.size() && (!string.is_ascii(i, ',')) &&
         (!string.is_ascii(i, ']')) && (!string.is_ascii(i, '#'))) {
    ++i;
  }
  il::ConstStringView value_string = string.substring(0, i);
  il::Type value_type = parse_type(value_string, il::io, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return ans;
  }

  switch (value_type) {
    case il::Type::null_t:
    case il::Type::bool_t:
    case il::Type::integer_t:
    case il::Type::double_t:
    case il::Type::string_t: {
      ans = parse_value_array(value_type, il::io, string, parse_status);
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        return ans;
      }
      status.set_ok();
      return ans;
    } break;
    case il::Type::array_t: {
      ans = parse_object_array(il::Type::array_t, '[', il::io, string,
                               parse_status);
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        return ans;
      }
      status.set_ok();
      return ans;
    }
    default:
      il::abort();
      return ans;
  }
}

il::Dynamic TomlParser::parse_value_array(il::Type value_type, il::io_t,
                                          il::ConstStringView& string,
                                          il::Status& status) {
  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Array<il::Dynamic>& array = ans.as_array();
  il::Status parse_status{};

  while (!string.is_empty() && (!string.is_ascii(0, ']'))) {
    il::Dynamic value = parse_value(il::io, string, parse_status);
    if (parse_status.not_ok()) {
      status = std::move(parse_status);
      return ans;
    }

    if (value.type() == value_type) {
      array.append(value);
    } else {
      status.set_error(il::Error::parse_heterogeneous_array);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return ans;
    }

    string = skip_whitespace_and_comments(string, il::io, parse_status);
    if (parse_status.not_ok()) {
      status = std::move(parse_status);
      return ans;
    }

    if (string.is_empty() || (!string.is_ascii(0, ','))) {
      break;
    }

    string.remove_prefix(1);
    string = skip_whitespace_and_comments(string, il::io, parse_status);
    if (parse_status.not_ok()) {
      status = std::move(parse_status);
      return ans;
    }
  }

  if (!string.is_empty()) {
    string.remove_prefix(1);
  }

  status.set_ok();
  return ans;
}

il::Dynamic TomlParser::parse_object_array(il::Type object_type, char delimiter,
                                           il::io_t,
                                           il::ConstStringView& string,
                                           il::Status& status) {
  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Array<il::Dynamic>& array = ans.as_array();
  il::Status parse_status{};

  while (!string.is_empty() && (!string.is_ascii(0, ']'))) {
    if (!string.is_ascii(0, delimiter)) {
      status.set_error(il::Error::parse_array);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return ans;
    }

    if (object_type == il::Type::array_t) {
      array.append(parse_array(il::io, string, parse_status));
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        return ans;
      }
    } else {
      il::abort();
    }

    string = il::remove_whitespace_left(string);
    if (string.is_empty() || (!string.is_ascii(0, ','))) {
      break;
    }
    string.remove_prefix(1);
    string = il::remove_whitespace_left(string);
  }

  if (string.is_empty() || (!string.is_ascii(0, ']'))) {
    status.set_error(il::Error::parse_array);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return ans;
  }
  string.remove_prefix(1);
  status.set_ok();
  return ans;
}

il::Dynamic TomlParser::parse_inline_table(il::io_t,
                                           il::ConstStringView& string,
                                           il::Status& status) {
  il::Dynamic ans = il::Dynamic{il::HashMapArray<il::String, il::Dynamic>{}};
  do {
    string.remove_prefix(1);
    if (string.is_empty()) {
      status.set_error(il::Error::parse_table);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return ans;
    }
    string = il::remove_whitespace_left(string);
    il::Status parse_status{};
    parse_key_value(il::io, string, ans.as_hash_map_array(), parse_status);
    if (parse_status.not_ok()) {
      status = std::move(parse_status);
      return ans;
    }
    string = il::remove_whitespace_left(string);
  } while (string.is_ascii(0, ','));

  if (string.is_empty() || (!string.is_ascii(0, '}'))) {
    status.set_error(il::Error::parse_table);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return ans;
  }

  string.remove_prefix(1);
  string = il::remove_whitespace_left(string);

  status.set_ok();
  return ans;
}

void TomlParser::parse_key_value(
    il::io_t, il::ConstStringView& string,
    il::HashMapArray<il::String, il::Dynamic>& toml, il::Status& status) {
  il::Status parse_status{};
  il::String key = parse_key('=', il::io, string, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return;
  }

  il::int_t i = toml.search(key);
  if (toml.found(i)) {
    status.set_error(il::Error::parse_duplicate_key);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return;
  }

  if (string.is_empty() || (!string.is_ascii(0, '='))) {
    status.set_error(il::Error::parse_key);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return;
  }
  string.remove_prefix(1);
  string = il::remove_whitespace_left(string);

  il::Dynamic value = parse_value(il::io, string, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return;
  }

  status.set_ok();
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
  if (string.is_ascii(0, '"')) {
    //////////////////////////
    // We have a key in a ".."
    //////////////////////////
    string.remove_prefix(1);
    while (string.size() > 0) {
      if (string.is_ascii(0, '\\')) {
        il::Status parse_status{};
        key.append(parse_escape_code(il::io, string, parse_status));
        if (parse_status.not_ok()) {
          status = std::move(parse_status);
          return key;
        }
      } else if (string.is_ascii(0, '"')) {
        string.remove_prefix(1);
        string = il::remove_whitespace_left(string);
        status.set_ok();
        return key;
      } else {
        // Check what's going on with unicode
        key.append(string.to_ascii(0));
        string.remove_prefix(1);
      }
    }
    status.set_error(il::Error::parse_string);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return key;
  } else {
    /////////////////////////////////////
    // We have a bare key: '..' or ->..<-
    /////////////////////////////////////
    // Search for the end of the key and move back to drop the whitespaces
    if (string.is_ascii(0, '\'')) {
      string.remove_prefix(1);
    }
    il::int_t j = 0;
    if (end != '@') {
      while (j < string.size() && (!string.is_ascii(j, end))) {
        ++j;
      }
    } else {
      while (j < string.size() && (!string.is_ascii(j, '.')) &&
             (!string.is_ascii(j, ']'))) {
        ++j;
      }
    }
    const il::int_t j_end = j;
    while (j > 0 &&
           (string.is_ascii(j - 1, ' ') || string.is_ascii(j - 1, '\t'))) {
      --j;
    }
    if (j == 0) {
      status.set_error(il::Error::parse_key);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return key;
    }

    // Check if the key contains forbidden characters
    il::ConstStringView key_string = string.substring(0, j);
    string.remove_prefix(j_end);

    for (il::int_t i = 0; i < key_string.size(); ++i) {
      if (key_string.is_ascii(i, ' ') || key_string.is_ascii(i, '\t')) {
        status.set_error(il::Error::parse_key);
        IL_SET_SOURCE(status);
        status.set_info("line", line_number_);
        return key;
      }
      if (key_string.is_ascii(i, '#')) {
        status.set_error(il::Error::parse_key);
        IL_SET_SOURCE(status);
        status.set_info("line", line_number_);
        return key;
      }
      if (key_string.is_ascii(i, '[') || key_string.is_ascii(i, ']')) {
        status.set_error(il::Error::parse_key);
        IL_SET_SOURCE(status);
        status.set_info("line", line_number_);
        return key;
      }
    }

    key = il::String{key_string.as_c_string(), j};
    status.set_ok();
    return key;
  }
}

il::Dynamic TomlParser::parse_value(il::io_t, il::ConstStringView& string,
                                    il::Status& status) {
  il::Dynamic ans{};

  // Check if there is a value
  if (string.is_empty() || string.is_ascii(0, '\n') ||
      string.is_ascii(0, '#')) {
    status.set_error(il::Error::parse_value);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return ans;
  }

  // Get the type of the value
  il::Status parse_status{};
  il::Type type = parse_type(string, il::io, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return ans;
  }

  switch (type) {
    case il::Type::bool_t:
      ans = parse_boolean(il::io, string, parse_status);
      break;
    case il::Type::integer_t:
    case il::Type::double_t:
      ans = parse_number(il::io, string, parse_status);
      break;
    case il::Type::string_t:
      ans = parse_string(il::io, string, parse_status);
      break;
    case il::Type::array_t:
      ans = parse_array(il::io, string, parse_status);
      break;
    case il::Type::hash_map_array_t:
      ans = parse_inline_table(il::io, string, parse_status);
      break;
    default:
      IL_UNREACHABLE;
  }

  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return ans;
  }

  status.set_ok();
  return ans;
}

void TomlParser::parse_table(il::io_t, il::ConstStringView& string,
                             il::HashMapArray<il::String, il::Dynamic>*& toml,
                             il::Status& status) {
  // Skip the '[' at the beginning of the table
  string.remove_prefix(1);

  if (string.is_empty()) {
    status.set_error(il::Error::parse_table);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return;
  } else if (string.is_ascii(0, '[')) {
    parse_table_array(il::io, string, toml, status);
    return;
  } else {
    parse_single_table(il::io, string, toml, status);
    return;
  }
}

void TomlParser::parse_single_table(
    il::io_t, il::ConstStringView& string,
    il::HashMapArray<il::String, il::Dynamic>*& toml, il::Status& status) {
  if (string.is_empty() || string.is_ascii(0, ']')) {
    status.set_error(il::Error::parse_table);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
  }

  il::String full_table_name{};
  bool inserted = false;
  while (!string.is_empty() && (!string.is_ascii(0, ']'))) {
    il::Status parse_status{};
    il::String table_name = parse_key('@', il::io, string, parse_status);
    if (parse_status.not_ok()) {
      status = std::move(parse_status);
      return;
    }
    if (table_name.is_empty()) {
      status.set_error(il::Error::parse_table);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return;
    }
    if (!full_table_name.is_empty()) {
      full_table_name.append('.');
    }
    full_table_name.append(table_name);

    il::int_t i = toml->search(table_name);
    if (toml->found(i)) {
      if (toml->value(i).is_hash_map_array()) {
        toml = &(toml->value(i).as_hash_map_array());
      } else if (toml->value(i).is_array()) {
        if (toml->value(i).as_array().size() > 0 &&
            toml->value(i).as_array().back().is_hash_map_array()) {
          toml = &(toml->value(i).as_array().back().as_hash_map_array());
        } else {
          status.set_error(il::Error::parse_duplicate_key);
          IL_SET_SOURCE(status);
          status.set_info("line", line_number_);
          return;
        }
      } else {
        status.set_error(il::Error::parse_duplicate_key);
        IL_SET_SOURCE(status);
        status.set_info("line", line_number_);
        return;
      }
    } else {
      inserted = true;
      toml->insert(table_name,
                   il::Dynamic{il::HashMapArray<il::String, il::Dynamic>{}},
                   il::io, i);
      toml = &(toml->value(i).as_hash_map_array());
    }

    string = il::remove_whitespace_left(string);
    while (!string.is_empty() && string.is_ascii(0, '.')) {
      string.remove_prefix(1);
    }
    string = il::remove_whitespace_left(string);
  }

  // TODO: One should check the redefinition of a table (line 1680)
  IL_UNUSED(inserted);

  string.remove_prefix(1);
  string = il::remove_whitespace_left(string);
  if (!string.is_empty() && (!string.is_ascii(0, '\n')) &&
      (!string.is_ascii(0, '#'))) {
    il::abort();
  }
  status.set_ok();
}

void TomlParser::parse_table_array(
    il::io_t, il::ConstStringView& string,
    il::HashMapArray<il::String, il::Dynamic>*& toml, il::Status& status) {
  string.remove_prefix(1);
  if (string.is_empty() || string.is_ascii(0, ']')) {
    status.set_error(il::Error::parse_table);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return;
  }

  il::String full_table_name{};
  while (!string.is_empty() && (!string.is_ascii(0, ']'))) {
    il::Status parse_status{};
    il::String table_name = parse_key('@', il::io, string, parse_status);
    if (parse_status.not_ok()) {
      status = std::move(parse_status);
      return;
    }
    if (table_name.is_empty()) {
      status.set_error(il::Error::parse_table);
      IL_SET_SOURCE(status);
      status.set_info("line", line_number_);
      return;
    }
    if (!full_table_name.is_empty()) {
      full_table_name.append('.');
    }
    full_table_name.append(table_name);

    string = il::remove_whitespace_left(string);
    il::int_t i = toml->search(table_name);
    if (toml->found(i)) {
      il::Dynamic& b = toml->value(i);
      if (!string.is_empty() && string.is_ascii(0, ']')) {
        if (!b.is_array()) {
          status.set_error(il::Error::parse_table);
          IL_SET_SOURCE(status);
          status.set_info("line", line_number_);
          return;
        }
        il::Array<il::Dynamic>& v = b.as_array();
        v.append(il::Dynamic{il::HashMapArray<il::String, il::Dynamic>{}});
        toml = &(v.back().as_hash_map_array());
      }
    } else {
      if (!string.is_empty() && string.is_ascii(0, ']')) {
        toml->insert(table_name, il::Dynamic{il::Array<il::Dynamic>{}}, il::io,
                     i);
        toml->value(i).as_array().append(
            il::Dynamic{il::HashMapArray<il::String, il::Dynamic>{}});
        toml = &(toml->value(i).as_array()[0].as_hash_map_array());
      } else {
        toml->insert(table_name,
                     il::Dynamic{il::HashMapArray<il::String, il::Dynamic>{}},
                     il::io, i);
        toml = &(toml->value(i).as_hash_map_array());
      }
    }
  }

  if (string.is_empty()) {
    status.set_error(il::Error::parse_table);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return;
  }
  string.remove_prefix(1);
  if (string.is_empty()) {
    status.set_error(il::Error::parse_table);
    IL_SET_SOURCE(status);
    status.set_info("line", line_number_);
    return;
  }
  string.remove_prefix(1);

  string = il::remove_whitespace_left(string);
  il::Status parse_status{};
  check_end_of_line_or_comment(string, il::io, parse_status);
  if (parse_status.not_ok()) {
    status = std::move(parse_status);
    return;
  }

  status.set_ok();
  return;
}

il::HashMapArray<il::String, il::Dynamic> TomlParser::parse(
    const il::String& filename, il::io_t, il::Status& status) {
  il::HashMapArray<il::String, il::Dynamic> root_toml{};
  il::HashMapArray<il::String, il::Dynamic>* pointer_toml = &root_toml;

#ifdef IL_UNIX
  file_ = std::fopen(filename.as_c_string(), "r+b");
#else
  il::UTF16String filename_utf16 = il::to_utf16(filename);
  file_ = _wfopen(filename_utf16.as_w_string(), L"r+b");
#endif
  if (!file_) {
    status.set_error(il::Error::filesystem_file_not_found);
    IL_SET_SOURCE(status);
    return root_toml;
  }

  line_number_ = 0;
  while (std::fgets(buffer_line_, max_line_length_ + 1, file_) != nullptr) {
    ++line_number_;

    il::ConstStringView line{buffer_line_};
    line = il::remove_whitespace_left(line);

    if (line.is_empty() || line.is_newline(0) || line.is_ascii(0, '#')) {
      continue;
    } else if (line.is_ascii(0, '[')) {
      pointer_toml = &root_toml;
      il::Status parse_status{};
      parse_table(il::io, line, pointer_toml, parse_status);
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        std::fclose(file_);
        return root_toml;
      }
    } else {
      il::Status parse_status{};
      parse_key_value(il::io, line, *pointer_toml, parse_status);
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        std::fclose(file_);
        return root_toml;
      }

      line = il::remove_whitespace_left(line);
      check_end_of_line_or_comment(line, il::io, parse_status);
      if (parse_status.not_ok()) {
        status = std::move(parse_status);
        return root_toml;
      }
    }
  }

  const int error = std::fclose(file_);
  if (error != 0) {
    status.set_error(il::Error::filesystem_cannot_close_file);
    IL_SET_SOURCE(status);
    return root_toml;
  }

  status.set_ok();
  return root_toml;
}
}  // namespace il
