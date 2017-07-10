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

#ifdef IL_WINDOWS
#include <il/UTF16String.h>
#include <il/unicode.h>
#endif

namespace il {

class TomlParser {
 private:
  static const il::int_t max_line_length_ = 200;
  char buffer_line_[max_line_length_ + 1];
  il::int_t line_number_;
  std::FILE *file_;

 public:
  TomlParser();
  il::MapArray<il::String, il::Dynamic> parse(const il::String &filename,
                                              il::io_t, il::Status &status);
  il::ConstStringView skipWhitespaceAndComments(il::ConstStringView string,
                                                il::io_t, il::Status &status);
  il::Dynamic parseValue(il::io_t, il::ConstStringView &string,
                         il::Status &status);
  il::Dynamic parseArray(il::io_t, il::ConstStringView &string,
                         il::Status &status);
  il::Dynamic parseValueArray(il::Type value_type, il::io_t,
                              il::ConstStringView &string, il::Status &status);
  il::Dynamic parseObjectArray(il::Type object_type, char delimiter, il::io_t,
                               il::ConstStringView &string, il::Status &status);
  il::Dynamic parseInlineTable(il::io_t, il::ConstStringView &string,
                               il::Status &status);
  void parseKeyValue(il::io_t, il::ConstStringView &string,
                     il::MapArray<il::String, il::Dynamic> &toml,
                     il::Status &status);
  void checkEndOfLineOrComment(il::ConstStringView string, il::io_t,
                               il::Status &status);
  il::String currentLine() const;
  il::Dynamic parseNumber(il::io_t, il::ConstStringView &string,
                          il::Status &status);
  il::Type parseType(il::ConstStringView string, il::io_t, il::Status &status);
  il::Dynamic parseBool(il::io_t, il::ConstStringView &string, Status &status);
  il::String parseStringLiteral(char delimiter, il::io_t,
                                il::ConstStringView &string,
                                il::Status &status);
  il::String parseEscapeCode(il::io_t, il::ConstStringView &string,
                             il::Status &status);
  il::String parseKey(char end, il::io_t, il::ConstStringView &string,
                      il::Status &status);
  void parseTable(il::io_t, il::ConstStringView &string,
                  il::MapArray<il::String, il::Dynamic> *&toml,
                  il::Status &status);
  void parseSingleTable(il::io_t, il::ConstStringView &string,
                        il::MapArray<il::String, il::Dynamic> *&toml,
                        il::Status &status);
  void parseTableArray(il::io_t, il::ConstStringView &string,
                       il::MapArray<il::String, il::Dynamic> *&toml,
                       il::Status &status);
  il::Dynamic parseString(il::io_t, il::ConstStringView &string,
                          il::Status &status);

 private:
  static bool containsDigit(char c);
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
      case il::Type::Bool: {
        if (array[j].toBool()) {
          const int error2 = std::fputs("true", file);
          IL_UNUSED(error2);
        } else {
          const int error2 = std::fputs("false", file);
          IL_UNUSED(error2);
        }
      } break;
      case il::Type::Integer: {
        const int error2 = std::fprintf(file, "%td", array[j].toInteger());
        IL_UNUSED(error2);
      } break;
      case il::Type::Double: {
        const int error2 = std::fprintf(file, "%e", array[j].toDouble());
        IL_UNUSED(error2);
      } break;
      case il::Type::String: {
        const int error2 = std::fputs("\"", file);
        const int error3 = std::fputs(array[j].asString().asCString(), file);
        const int error4 = std::fputs("\"", file);
        IL_UNUSED(error2);
        IL_UNUSED(error3);
        IL_UNUSED(error4);
      } break;
      case il::Type::Array: {
        save_array(array[j].asArray(), il::io, file, status);
        if (!status.ok()) {
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
  status.setOk();
}

template <typename M>
inline void save_aux(const M &toml, const il::String &name, il::io_t,
                     std::FILE *file, il::Status &status) {
  int error = 0;
  // Add an object that sets the error on destruction

  for (il::int_t i = toml.first(); i != toml.sentinel(); i = toml.next(i)) {
    const il::Dynamic &value = toml.value(i);
    const il::Type type = value.type();
    if (type != il::Type::MapArray && type != il::Type::Map) {
      error = std::fputs(toml.key(i).asCString(), file);
      if (error == EOF) return;
      error = std::fputs(" = ", file);
      if (error == EOF) return;
      switch (type) {
        case il::Type::Bool:
          if (value.toBool()) {
            error = std::fputs("true", file);
          } else {
            error = std::fputs("false", file);
          }
          if (error == EOF) return;
          break;
        case il::Type::Integer:
          error = std::fprintf(file, "%td", value.toInteger());
          if (error == EOF) return;
          break;
        case il::Type::Double:
          error = std::fprintf(file, "%e", value.toDouble());
          if (error == EOF) return;
          break;
        case il::Type::String:
          error = std::fputs("\"", file);
          if (error == EOF) return;
          error = std::fputs(value.asString().asCString(), file);
          if (error == EOF) return;
          error = std::fputs("\"", file);
          if (error == EOF) return;
          break;
        case il::Type::ArrayOfDouble: {
          const il::Array<double> &v = value.asArrayOfDouble();
          error = std::fputs("[ ", file);
          if (error == EOF) return;
          for (il::int_t i = 0; i < v.size(); ++i) {
            std::fprintf(file, "%e", v[i]);
            if (i + 1 < v.size()) {
              error = std::fputs(", ", file);
              if (error == EOF) return;
            }
          }
          error = std::fputs(" ]", file);
          if (error == EOF) return;
        } break;
        case il::Type::Array2dOfDouble: {
          const il::Array2D<double> &v = value.asArray2dOfDouble();
          error = std::fputs("[ ", file);
          if (error == EOF) return;
          for (il::int_t i = 0; i < v.size(0); ++i) {
            error = std::fputs("[ ", file);
            if (error == EOF) return;
            for (il::int_t j = 0; j < v.size(1); ++j) {
              std::fprintf(file, "%e", v(i, j));
              if (j + 1 < v.size(1)) {
                error = std::fputs(", ", file);
                if (error == EOF) return;
              }
            }
            error = std::fputs(" ]", file);
            if (error == EOF) return;
            if (i + 1 < v.size(0)) {
              error = std::fputs(", ", file);
              if (error == EOF) return;
            }
          }
          error = std::fputs(" ]", file);
          if (error == EOF) return;
        } break;
        case il::Type::Array: {
          save_array(value.asArray(), il::io, file, status);
          status.abortOnError();
        } break;
        default:
          IL_UNREACHABLE;
      }
      error = std::fputs("\n", file);
      if (error == EOF) return;
    } else if (type == il::Type::MapArray) {
      error = std::fputs("\n[", file);
      if (error == EOF) return;
      if (name.size() != 0) {
        error = std::fputs(name.asCString(), file);
        if (error == EOF) return;
        error = std::fputs(".", file);
        if (error == EOF) return;
      }
      error = std::fputs(toml.key(i).asCString(), file);
      if (error == EOF) return;
      error = std::fputs("]\n", file);
      if (error == EOF) return;
      save_aux(value.asMapArray(), toml.key(i), il::io, file, status);
      if (!status.ok()) {
        status.rearm();
        return;
      }
    } else if (type == il::Type::Map) {
      error = std::fputs("\n[", file);
      if (error == EOF) return;
      if (name.size() != 0) {
        error = std::fputs(name.asCString(), file);
        if (error == EOF) return;
        error = std::fputs(".", file);
        if (error == EOF) return;
      }
      error = std::fputs(toml.key(i).asCString(), file);
      if (error == EOF) return;
      error = std::fputs("]\n", file);
      if (error == EOF) return;
      save_aux(value.asMap(), toml.key(i), il::io, file, status);
      if (!status.ok()) {
        status.rearm();
        return;
      }
    }
  }

  status.setOk();
  return;
}

template <>
class SaveHelper<il::Map<il::String, il::Dynamic>> {
 public:
  static void save(const il::Map<il::String, il::Dynamic> &toml,
                   const il::String &filename, il::io_t, il::Status &status) {
#ifdef IL_UNIX
    std::FILE *file = std::fopen(filename.asCString(), "wb");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE *file = _wfopen(filename_utf16.asWString(), L"wb");
#endif
    if (!file) {
      status.setError(il::Error::FilesystemFileNotFound);
      return;
    }

    il::String root_name{};
    save_aux(toml, root_name, il::io, file, status);
    if (!status.ok()) {
      status.rearm();
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.setError(il::Error::FilesystemCanNotCloseFile);
      return;
    }

    status.setOk();
    return;
  }
};

template <>
class SaveHelperToml<il::MapArray<il::String, il::Dynamic>> {
 public:
  static void save(const il::MapArray<il::String, il::Dynamic> &toml,
                   const il::String &filename, il::io_t, il::Status &status) {
#ifdef IL_UNIX
    std::FILE *file = std::fopen(filename.asCString(), "wb");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE *file = _wfopen(filename_utf16.asWString(), L"wb");
#endif
    if (!file) {
      status.setError(il::Error::FilesystemFileNotFound);
      return;
    }

    il::String root_name{};
    save_aux(toml, root_name, il::io, file, status);
    if (!status.ok()) {
      status.rearm();
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.setError(il::Error::FilesystemCanNotCloseFile);
      return;
    }

    status.setOk();
    return;
  }
};

il::MapArray<il::String, il::Dynamic> parse(const il::String &filename,
                                            il::io_t, il::Status &status);

template <>
class LoadHelperToml<il::MapArray<il::String, il::Dynamic>> {
 public:
  static il::MapArray<il::String, il::Dynamic> load(const il::String &filename,
                                                    il::io_t,
                                                    il::Status &status) {
    il::TomlParser parser{};
    return parser.parse(filename, il::io, status);
  }
};
}  // namespace il

#endif  // IL_TOML_H