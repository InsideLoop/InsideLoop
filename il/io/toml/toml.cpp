//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#include <il/container/string/algorithm_string.h>
#include <il/io/toml/toml.h>

namespace il {

TomlParser::TomlParser() {}

il::StringView TomlParser::skipWhitespaceAndComments(il::StringView string,
                                                     il::io_t,
                                                     il::Status& status) {
  string.trimPrefix();
  while (string.isEmpty() || string[0] == '#' || string.startsWithNewLine()) {
    const char* error = std::fgets(buffer_line_, max_line_length_ + 1, file_);
    if (error == nullptr) {
      IL_SET_SOURCE(status);
      status.setError(il::Error::ParseUnclosedArray);
      status.setInfo("line", line_number_);
      return string;
    }
    ++line_number_;
    string = il::StringView{buffer_line_};
    string.trimPrefix();
  }

  status.setOk();
  return string;
}

bool TomlParser::containsDigit(char c) { return c >= '0' && c <= '9'; }

void TomlParser::checkEndOfLineOrComment(il::StringView string, il::io_t,
                                         il::Status& status) {
  if (!string.isEmpty() && (!string.startsWithNewLine()) &&
      (string[0] != '#')) {
    IL_SET_SOURCE(status);
    status.setError(il::Error::ParseUnidentifiedTrailingCharacter);
    status.setInfo("line", line_number_);
  } else {
    status.setOk();
  }
}

il::String TomlParser::currentLine() const {
  return il::toString(line_number_);
}

il::Type TomlParser::parseType(il::StringView string, il::io_t,
                               il::Status& status) {
  if (string[0] == '"' || string[0] == '\'') {
    status.setOk();
    return il::Type::String;
  } else if (string.startsWithDigit() || string[0] == '-' || string[0] == '+') {
    il::int_t i = 0;
    if (string[0] == '-' || string[0] == '+') {
      ++i;
    }
    while (i < string.size() && string.hasDigit(i)) {
      ++i;
    }
    if (i < string.size() && string[i] == '.') {
      ++i;
      while (i < string.size() && string.hasDigit(i)) {
        ++i;
      }
      status.setOk();
      return il::Type::Double;
    } else {
      status.setOk();
      return il::Type::Integer;
    }
  } else if (string[0] == 't' || string[0] == 'f') {
    status.setOk();
    return il::Type::Bool;
  } else if (string[0] == '[') {
    status.setOk();
    return il::Type::Array;
  } else if (string[0] == '{') {
    status.setOk();
    return il::Type::MapArray;
  } else {
    status.setError(il::Error::ParseCanNotDetermineType);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return il::Type::Null;
  }
}

il::Dynamic TomlParser::parseBool(il::io_t, il::StringView& string,
                                  Status& status) {
  if (string.size() > 3 && string[0] == 't' && string[1] == 'r' &&
      string[2] == 'u' && string[3] == 'e') {
    status.setOk();
    string.removePrefix(4);
    return il::Dynamic{true};
  } else if (string.size() > 4 && string[0] == 'f' && string[1] == 'a' &&
             string[2] == 'l' && string[3] == 's' && string[4] == 'e') {
    status.setOk();
    string.removePrefix(5);
    return il::Dynamic{false};
  } else {
    status.setError(il::Error::ParseBool);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return il::Dynamic{};
  }
}

il::Dynamic TomlParser::parseNumber(il::io_t, il::StringView& string,
                                    il::Status& status) {
  // Skip the +/- sign at the beginning of the string
  il::int_t i = 0;
  if (i < string.size() && (string[i] == '+' || string[i] == '-')) {
    ++i;
  }

  // Check that there is no leading 0
  if (i + 1 < string.size() && string[i] == '0' &&
      !(string[i + 1] == '.' || string[i + 1] == ' ' ||
        string.hasNewLine(i + 1))) {
    status.setError(il::Error::ParseNumber);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return il::Dynamic{};
  }

  // Skip the digits (there should be at least one) before the '.' ot the 'e'
  const il::int_t i_begin_number = i;
  while (i < string.size() && string.hasDigit(i)) {
    ++i;
    if (i < string.size() && string[i] == '_') {
      ++i;
      if (i == string.size() || (!string.hasDigit(i + 1))) {
        status.setError(il::Error::ParseNumber);
        IL_SET_SOURCE(status);
        status.setInfo("line", line_number_);
        return il::Dynamic{};
      }
    }
  }
  if (i == i_begin_number) {
    status.setError(il::Error::ParseNumber);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
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
      status.setError(il::Error::ParseDouble);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
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
    while (i < string.size() && string.hasDigit(i)) {
      ++i;
    }
    if (i == i_begin_number) {
      status.setError(il::Error::ParseDouble);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
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
      while (i < string.size() && string.hasDigit(i)) {
        ++i;
      }
      if (i == i_begin_exponent) {
        status.setError(il::Error::ParseDouble);
        IL_SET_SOURCE(status);
        status.setInfo("line", line_number_);
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
    if (string[j] != '_') {
      number.append(string[j]);
    }
  }
  il::StringView view{il::StringType::Bytes, number.data(), i};

  if (is_float) {
    status.setOk();
    string.removePrefix(i);
    double x = std::atof(view.asCString());
    return il::Dynamic{x};
  } else {
    status.setOk();
    string.removePrefix(i);
    il::int_t n = std::atoll(view.asCString());
    return il::Dynamic{n};
  }
}

il::Dynamic TomlParser::parseString(il::io_t, il::StringView& string,
                                    il::Status& status) {
  IL_EXPECT_FAST(string[0] == '"' || string[0] == '\'');

  const char delimiter = string[0];

  il::Status parse_status{};
  il::Dynamic ans = parseStringLiteral(delimiter, il::io, string, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return ans;
  }

  status.setOk();
  return ans;
}

il::String TomlParser::parseStringLiteral(char delimiter, il::io_t,
                                          il::StringView& string,
                                          il::Status& status) {
  il::String ans{};

  string.removePrefix(1);
  while (!string.isEmpty()) {
    if (delimiter == '"' && string[0] == '\\') {
      il::Status parse_status{};
      ans.append(parseEscapeCode(il::io, string, parse_status));
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        return ans;
      }
    } else if (string[0] == delimiter) {
      string.removePrefix(1);
      string = il::removeWhitespaceLeft(string);
      status.setOk();
      return ans;
    } else {
      // TODO: I am not sure what will happen with a Unicode string
      ans.append(string.rune(0));
      string.removePrefix(string.nextRune(0));
    }
  }

  status.setError(il::Error::ParseString);
  IL_SET_SOURCE(status);
  status.setInfo("line", line_number_);
  return ans;
}

il::String TomlParser::parseEscapeCode(il::io_t, il::StringView& string,
                                       il::Status& status) {
  IL_EXPECT_FAST(string.size() > 0 && string[0] == '\\');

  il::String ans{};
  il::int_t i = 1;
  if (i == string.size()) {
    status.setError(il::Error::ParseString);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
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
    case 'U': {
      status.setError(il::Error::ParseString);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return ans;
    } break;
    default:
      status.setError(il::Error::ParseString);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return ans;
  }

  string.removePrefix(2);
  ans.append(value);
  status.setOk();
  return ans;
}

il::Dynamic TomlParser::parseArray(il::io_t, il::StringView& string,
                                   il::Status& status) {
  IL_EXPECT_FAST(!string.isEmpty() && string[0] == '[');

  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Status parse_status{};

  string.removePrefix(1);
  string = skipWhitespaceAndComments(string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return ans;
  }

  if (!string.isEmpty() && string[0] == ']') {
    string.removePrefix(1);
    status.setOk();
    return ans;
  }

  il::int_t i = 0;
  while (i < string.size() && (string[i] != ',') && (string[i] != ']') &&
         (string[i] != '#')) {
    ++i;
  }
  il::StringView value_string = string.subview(0, i);
  il::Type value_type = parseType(value_string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return ans;
  }

  switch (value_type) {
    case il::Type::Null:
    case il::Type::Bool:
    case il::Type::Integer:
    case il::Type::Double:
    case il::Type::String: {
      ans = parseValueArray(value_type, il::io, string, parse_status);
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        return ans;
      }
      status.setOk();
      return ans;
    } break;
    case il::Type::Array: {
      ans =
          parseObjectArray(il::Type::Array, '[', il::io, string, parse_status);
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        return ans;
      }
      status.setOk();
      return ans;
    }
    default:
      il::abort();
      return ans;
  }
}

il::Dynamic TomlParser::parseValueArray(il::Type value_type, il::io_t,
                                        il::StringView& string,
                                        il::Status& status) {
  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Array<il::Dynamic>& array = ans.asArray();
  il::Status parse_status{};

  while (!string.isEmpty() && (string[0] != ']')) {
    il::Dynamic value = parseValue(il::io, string, parse_status);
    if (!parse_status.ok()) {
      status = std::move(parse_status);
      return ans;
    }

    if (value.type() == value_type) {
      array.append(value);
    } else {
      status.setError(il::Error::ParseHeterogeneousArray);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return ans;
    }

    string = skipWhitespaceAndComments(string, il::io, parse_status);
    if (!parse_status.ok()) {
      status = std::move(parse_status);
      return ans;
    }

    if (string.isEmpty() || (string[0] != ',')) {
      break;
    }

    string.removePrefix(1);
    string = skipWhitespaceAndComments(string, il::io, parse_status);
    if (!parse_status.ok()) {
      status = std::move(parse_status);
      return ans;
    }
  }

  if (!string.isEmpty()) {
    string.removePrefix(1);
  }

  status.setOk();
  return ans;
}

il::Dynamic TomlParser::parseObjectArray(il::Type object_type, char delimiter,
                                         il::io_t, il::StringView& string,
                                         il::Status& status) {
  il::Dynamic ans{il::Array<il::Dynamic>{}};
  il::Array<il::Dynamic>& array = ans.asArray();
  il::Status parse_status{};

  while (!string.isEmpty() && (string[0] != ']')) {
    if (string[0] != delimiter) {
      status.setError(il::Error::ParseArray);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return ans;
    }

    if (object_type == il::Type::Array) {
      array.append(parseArray(il::io, string, parse_status));
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        return ans;
      }
    } else {
      il::abort();
    }

    string = il::removeWhitespaceLeft(string);
    if (string.isEmpty() || (string[0] != ',')) {
      break;
    }
    string.removePrefix(1);
    string = il::removeWhitespaceLeft(string);
  }

  if (string.isEmpty() || (string[0] != ']')) {
    status.setError(il::Error::ParseArray);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return ans;
  }
  string.removePrefix(1);
  status.setOk();
  return ans;
}

il::Dynamic TomlParser::parseInlineTable(il::io_t, il::StringView& string,
                                         il::Status& status) {
  il::Dynamic ans = il::Dynamic{il::MapArray<il::String, il::Dynamic>{}};
  do {
    string.removePrefix(1);
    if (string.isEmpty()) {
      status.setError(il::Error::ParseTable);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return ans;
    }
    string = il::removeWhitespaceLeft(string);
    il::Status parse_status{};
    parseKeyValue(il::io, string, ans.asMapArray(), parse_status);
    if (!parse_status.ok()) {
      status = std::move(parse_status);
      return ans;
    }
    string = il::removeWhitespaceLeft(string);
  } while (string[0] == ',');

  if (string.isEmpty() || (string[0] != '}')) {
    status.setError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return ans;
  }

  string.removePrefix(1);
  string = il::removeWhitespaceLeft(string);

  status.setOk();
  return ans;
}

void TomlParser::parseKeyValue(il::io_t, il::StringView& string,
                               il::MapArray<il::String, il::Dynamic>& toml,
                               il::Status& status) {
  il::Status parse_status{};
  il::String key = parseKey('=', il::io, string, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return;
  }

  il::int_t i = toml.search(key);
  if (toml.found(i)) {
    status.setError(il::Error::ParseDuplicateKey);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return;
  }

  if (string.isEmpty() || (string[0] != '=')) {
    status.setError(il::Error::ParseKey);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return;
  }
  string.removePrefix(1);
  string = il::removeWhitespaceLeft(string);

  il::Dynamic value = parseValue(il::io, string, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return;
  }

  status.setOk();
  toml.set(key, value);
}

il::String TomlParser::parseKey(char end, il::io_t, il::StringView& string,
                                il::Status& status) {
  IL_EXPECT_FAST(end == '=' || end == '"' || end == '\'' || end == '@');

  il::String key{};
  string = il::removeWhitespaceLeft(string);
  IL_EXPECT_FAST(string.size() > 0);

  il::Status parse_status{};
  if (string[0] == '"') {
    //////////////////////////
    // We have a key in a ".."
    //////////////////////////
    string.removePrefix(1);
    while (string.size() > 0) {
      if (string[0] == '\\') {
        il::Status parse_status{};
        key.append(parseEscapeCode(il::io, string, parse_status));
        if (!parse_status.ok()) {
          status = std::move(parse_status);
          return key;
        }
      } else if (string[0] == '"') {
        string.removePrefix(1);
        string = il::removeWhitespaceLeft(string);
        status.setOk();
        return key;
      } else {
        // Check what's going on with unicode
        key.append(string[0]);
        string.removePrefix(1);
      }
    }
    status.setError(il::Error::ParseString);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return key;
  } else {
    /////////////////////////////////////
    // We have a bare key: '..' or ->..<-
    /////////////////////////////////////
    // Search for the end of the key and move back to drop the whitespaces
    if (string[0] == '\'') {
      string.removePrefix(1);
    }
    il::int_t j = 0;
    if (end != '@') {
      while (j < string.size() && (string[j] != end)) {
        ++j;
      }
    } else {
      while (j < string.size() && (string[j] != '.') && (string[j] != ']')) {
        ++j;
      }
    }
    const il::int_t j_end = j;
    while (j > 0 && (string[j - 1] == ' ' || string[j - 1] == '\t')) {
      --j;
    }
    if (j == 0) {
      status.setError(il::Error::ParseKey);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return key;
    }

    // Check if the key has forbidden characters
    il::StringView key_string = string.subview(0, j);
    string.removePrefix(j_end);

    for (il::int_t i = 0; i < key_string.size(); ++i) {
      if (key_string[i] == ' ' || key_string[i] == '\t') {
        status.setError(il::Error::ParseKey);
        IL_SET_SOURCE(status);
        status.setInfo("line", line_number_);
        return key;
      }
      if (key_string[i] == '#') {
        status.setError(il::Error::ParseKey);
        IL_SET_SOURCE(status);
        status.setInfo("line", line_number_);
        return key;
      }
      if (key_string[i] == '[' || key_string[i] == ']') {
        status.setError(il::Error::ParseKey);
        IL_SET_SOURCE(status);
        status.setInfo("line", line_number_);
        return key;
      }
    }

    key = il::String{il::StringType::Bytes, key_string.data(), j};
    status.setOk();
    return key;
  }
}

il::Dynamic TomlParser::parseValue(il::io_t, il::StringView& string,
                                   il::Status& status) {
  il::Dynamic ans{};

  // Check if there is a value
  if (string.isEmpty() || string[0] == '\n' || string[0] == '#') {
    status.setError(il::Error::ParseValue);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return ans;
  }

  // Get the type of the value
  il::Status parse_status{};
  il::Type type = parseType(string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return ans;
  }

  switch (type) {
    case il::Type::Bool:
      ans = parseBool(il::io, string, parse_status);
      break;
    case il::Type::Integer:
    case il::Type::Double:
      ans = parseNumber(il::io, string, parse_status);
      break;
    case il::Type::String:
      ans = parseString(il::io, string, parse_status);
      break;
    case il::Type::Array:
      ans = parseArray(il::io, string, parse_status);
      break;
    case il::Type::MapArray:
      ans = parseInlineTable(il::io, string, parse_status);
      break;
    default:
      IL_UNREACHABLE;
  }

  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return ans;
  }

  status.setOk();
  return ans;
}

void TomlParser::parseTable(il::io_t, il::StringView& string,
                            il::MapArray<il::String, il::Dynamic>*& toml,
                            il::Status& status) {
  // Skip the '[' at the beginning of the table
  string.removePrefix(1);

  if (string.isEmpty()) {
    status.setError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return;
  } else if (string[0] == '[') {
    parseTableArray(il::io, string, toml, status);
    return;
  } else {
    parseSingleTable(il::io, string, toml, status);
    return;
  }
}

void TomlParser::parseSingleTable(il::io_t, il::StringView& string,
                                  il::MapArray<il::String, il::Dynamic>*& toml,
                                  il::Status& status) {
  if (string.isEmpty() || string[0] == ']') {
    status.setError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
  }

  il::String full_table_name{};
  bool inserted = false;
  while (!string.isEmpty() && (string[0] != ']')) {
    il::Status parse_status{};
    il::String table_name = parseKey('@', il::io, string, parse_status);
    if (!parse_status.ok()) {
      status = std::move(parse_status);
      return;
    }
    if (table_name.isEmpty()) {
      status.setError(il::Error::ParseTable);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return;
    }
    if (!full_table_name.isEmpty()) {
      full_table_name.append('.');
    }
    full_table_name.append(table_name);

    il::int_t i = toml->search(table_name);
    if (toml->found(i)) {
      if (toml->value(i).isMapArray()) {
        toml = &(toml->value(i).asMapArray());
      } else if (toml->value(i).isArray()) {
        if (toml->value(i).asArray().size() > 0 &&
            toml->value(i).asArray().back().isMapArray()) {
          toml = &(toml->value(i).asArray().back().asMapArray());
        } else {
          status.setError(il::Error::ParseDuplicateKey);
          IL_SET_SOURCE(status);
          status.setInfo("line", line_number_);
          return;
        }
      } else {
        status.setError(il::Error::ParseDuplicateKey);
        IL_SET_SOURCE(status);
        status.setInfo("line", line_number_);
        return;
      }
    } else {
      inserted = true;
      toml->insert(table_name,
                   il::Dynamic{il::MapArray<il::String, il::Dynamic>{}}, il::io,
                   i);
      toml = &(toml->value(i).asMapArray());
    }

    string = il::removeWhitespaceLeft(string);
    while (!string.isEmpty() && string[0] == '.') {
      string.removePrefix(1);
    }
    string = il::removeWhitespaceLeft(string);
  }

  // TODO: One should check the redefinition of a table (line 1680)
  IL_UNUSED(inserted);

  string.removePrefix(1);
  string = il::removeWhitespaceLeft(string);
  if (!string.isEmpty() && (string[0] != '\n') && (string[0] != '#')) {
    il::abort();
  }
  status.setOk();
}

void TomlParser::parseTableArray(il::io_t, il::StringView& string,
                                 il::MapArray<il::String, il::Dynamic>*& toml,
                                 il::Status& status) {
  string.removePrefix(1);
  if (string.isEmpty() || string[0] == ']') {
    status.setError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return;
  }

  il::String full_table_name{};
  while (!string.isEmpty() && (string[0] != ']')) {
    il::Status parse_status{};
    il::String table_name = parseKey('@', il::io, string, parse_status);
    if (!parse_status.ok()) {
      status = std::move(parse_status);
      return;
    }
    if (table_name.isEmpty()) {
      status.setError(il::Error::ParseTable);
      IL_SET_SOURCE(status);
      status.setInfo("line", line_number_);
      return;
    }
    if (!full_table_name.isEmpty()) {
      full_table_name.append('.');
    }
    full_table_name.append(table_name);

    string = il::removeWhitespaceLeft(string);
    il::int_t i = toml->search(table_name);
    if (toml->found(i)) {
      il::Dynamic& b = toml->value(i);
      if (!string.isEmpty() && string[0] == ']') {
        if (!b.isArray()) {
          status.setError(il::Error::ParseTable);
          IL_SET_SOURCE(status);
          status.setInfo("line", line_number_);
          return;
        }
        il::Array<il::Dynamic>& v = b.asArray();
        v.append(il::Dynamic{il::MapArray<il::String, il::Dynamic>{}});
        toml = &(v.back().asMapArray());
      }
    } else {
      if (!string.isEmpty() && string[0] == ']') {
        toml->insert(table_name, il::Dynamic{il::Array<il::Dynamic>{}}, il::io,
                     i);
        toml->value(i).asArray().append(
            il::Dynamic{il::MapArray<il::String, il::Dynamic>{}});
        toml = &(toml->value(i).asArray()[0].asMapArray());
      } else {
        toml->insert(table_name,
                     il::Dynamic{il::MapArray<il::String, il::Dynamic>{}},
                     il::io, i);
        toml = &(toml->value(i).asMapArray());
      }
    }
  }

  if (string.isEmpty()) {
    status.setError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return;
  }
  string.removePrefix(1);
  if (string.isEmpty()) {
    status.setError(il::Error::ParseTable);
    IL_SET_SOURCE(status);
    status.setInfo("line", line_number_);
    return;
  }
  string.removePrefix(1);

  string = il::removeWhitespaceLeft(string);
  il::Status parse_status{};
  checkEndOfLineOrComment(string, il::io, parse_status);
  if (!parse_status.ok()) {
    status = std::move(parse_status);
    return;
  }

  status.setOk();
  return;
}

il::MapArray<il::String, il::Dynamic> TomlParser::parse(
    const il::String& filename, il::io_t, il::Status& status) {
  il::MapArray<il::String, il::Dynamic> root_toml{};
  il::MapArray<il::String, il::Dynamic>* pointer_toml = &root_toml;

#ifdef IL_UNIX
  file_ = std::fopen(filename.asCString(), "r+b");
#else
  il::UTF16String filename_utf16 = il::toUtf16(filename);
  file_ = _wfopen(filename_utf16.asWString(), L"r+b");
#endif
  if (!file_) {
    status.setError(il::Error::FilesystemFileNotFound);
    IL_SET_SOURCE(status);
    return root_toml;
  }

  line_number_ = 0;
  while (std::fgets(buffer_line_, max_line_length_ + 1, file_) != nullptr) {
    ++line_number_;

    il::StringView line{il::StringType::Bytes, buffer_line_,
                        il::size(buffer_line_)};
    line = il::removeWhitespaceLeft(line);

    if (line.isEmpty() || line.startsWithNewLine() || line[0] == '#') {
      continue;
    } else if (line[0] == '[') {
      pointer_toml = &root_toml;
      il::Status parse_status{};
      parseTable(il::io, line, pointer_toml, parse_status);
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        std::fclose(file_);
        return root_toml;
      }
    } else {
      il::Status parse_status{};
      parseKeyValue(il::io, line, *pointer_toml, parse_status);
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        std::fclose(file_);
        return root_toml;
      }

      line = il::removeWhitespaceLeft(line);
      checkEndOfLineOrComment(line, il::io, parse_status);
      if (!parse_status.ok()) {
        status = std::move(parse_status);
        return root_toml;
      }
    }
  }

  const int error = std::fclose(file_);
  if (error != 0) {
    status.setError(il::Error::FilesystemCanNotCloseFile);
    IL_SET_SOURCE(status);
    return root_toml;
  }

  status.setOk();
  return root_toml;
}
}  // namespace il
