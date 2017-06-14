//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_FILEPACK_H
#define IL_FILEPACK_H

#include <cstdio>

#include <il/Array.h>
#include <il/HashMap.h>
#include <il/HashMapArray.h>
#include <il/String.h>
#include <il/container/dynamic/Dynamic.h>
#include <il/io/io_base.h>

#ifdef IL_WINDOWS
#include <il/unicode.h>
#include <il/UTF16String.h>
#endif

namespace il {

il::int_t read_varint(il::io_t, il::int_t& k, std::FILE* file) {
  std::size_t ans = 0;
  std::size_t multiplier = 1;
  const std::size_t max_byte = 1 << 7;
  const unsigned char continuation_mask = 0x7F;
  const unsigned char continuation_byte = 0x80;
  unsigned char byte;
  do {
    std::fread(&byte, sizeof(unsigned char), 1, file);
    k += 1;
    ans += multiplier * (byte & continuation_mask);
    multiplier *= max_byte;
  } while ((byte & continuation_byte) == continuation_byte);

  return static_cast<il::int_t>(ans);
}

void write_varint(il::int_t n, il::io_t, il::int_t& k, std::FILE* file) {
  std::size_t un = static_cast<std::size_t>(n);

  const std::size_t max_byte = 1 << 7;
  const unsigned char continuation_byte = 0x80;
  while (true) {
    unsigned char r = static_cast<unsigned char>(un % max_byte);
    un /= max_byte;
    if (un == 0) {
      std::fwrite(&r, sizeof(unsigned char), 1, file);
      k += 1;
      break;
    } else {
      r |= continuation_byte;
      std::fwrite(&r, sizeof(unsigned char), 1, file);
      k += 1;
    }
  }
}

void aux_load(il::int_t n, il::io_t,
              il::HashMap<il::String, il::Dynamic>& config, std::FILE* file) {
  IL_UNUSED(n);
  il::int_t k = 0;

  while (true) {
    il::int_t size_string = read_varint(il::io, k, file);
    if (size_string == 0) {
      break;
    }
    il::Array<char> raw_string{size_string - 1};
    std::fread(raw_string.data(), sizeof(char), size_string - 1, file);
    il::String string{raw_string.data(), size_string - 1};
    k += string.size() + 1;

    il::Type type;
    fread(&type, sizeof(il::Type), 1, file);
    k += sizeof(il::Type);

    switch (type) {
      case il::Type::bool_t: {
        bool value = false;
        std::fread(&value, sizeof(bool), 1, file);
        k += sizeof(bool);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::integer_t: {
        const il::int_t value = read_varint(il::io, k, file);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::float_t: {
        float value = 0.0f;
        std::fread(&value, sizeof(float), 1, file);
        k += sizeof(float);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::double_t: {
        double value = 0.0;
        std::fread(&value, sizeof(double), 1, file);
        k += sizeof(double);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::string_t: {
        il::int_t size = 0;
        std::fread(&size, sizeof(il::int_t), 1, file);
        k += sizeof(il::int_t);
        il::Array<char> raw_value{size + 1};
        std::fread(raw_value.data(), sizeof(char), size + 1, file);
        k += size + 1;
        il::String value = raw_value.data();
        config.set(std::move(string), il::Dynamic{std::move(value)});
      } break;
      case il::Type::array_of_double_t: {
        il::int_t size = 0;
        std::fread(&size, sizeof(il::int_t), 1, file);
        k += sizeof(il::int_t);
        il::Array<double> v{size};
        std::fread(v.data(), sizeof(double), size, file);
        k += sizeof(double) * size;
        config.set(std::move(string), il::Dynamic{std::move(v)});
      } break;
      case il::Type::array2d_of_uint8_t: {
        il::int_t size0 = 0;
        il::int_t size1 = 0;
        std::fread(&size0, sizeof(il::int_t), 1, file);
        std::fread(&size1, sizeof(il::int_t), 1, file);
        k += 2 * sizeof(il::int_t);
        il::Array2D<unsigned char> A{size0, size1};
        std::fread(A.data(), sizeof(double), size0 * size1, file);
        k += sizeof(unsigned char) * size0 * size1;
        config.set(std::move(string), il::Dynamic{std::move(A)});
      } break;
      case il::Type::array2d_of_double_t: {
        il::int_t size0 = 0;
        il::int_t size1 = 0;
        std::fread(&size0, sizeof(il::int_t), 1, file);
        std::fread(&size1, sizeof(il::int_t), 1, file);
        k += 2 * sizeof(il::int_t);
        il::Array2D<double> v{size0, size1};
        std::fread(v.data(), sizeof(double), size0 * size1, file);
        k += sizeof(double) * size0 * size1;
        config.set(std::move(string), il::Dynamic{std::move(v)});
      } break;
      case il::Type::hash_map_array_t:
      case il::Type::hash_map_t: {
        il::int_t n_map = 0;
        const il::int_t n = read_varint(il::io, n_map, file);
        il::int_t size = 0;
        il::int_t i = config.search(string);
        config.insert(std::move(string),
                      il::Dynamic{il::HashMap<il::String, il::Dynamic>{n}},
                      il::io, i);
        if (!config.found(i)) {
        }
        il::HashMap<il::String, il::Dynamic>& config_inner =
            config.value(i).as_hash_map();
        aux_load(size, il::io, config_inner, file);
        k += size;
      } break;
      default:
        IL_UNREACHABLE;
    }
  }
}

void aux_load(il::int_t n, il::io_t,
              il::HashMapArray<il::String, il::Dynamic>& config,
              std::FILE* file) {
  IL_UNUSED(n);
  il::int_t k = 0;

  while (true) {
    il::int_t size_string = read_varint(il::io, k, file);
    if (size_string == 0) {
      break;
    }
    il::Array<char> raw_string{size_string - 1};
    std::fread(raw_string.data(), sizeof(char), size_string - 1, file);
    il::String string{raw_string.data(), size_string - 1};
    k += string.size() + 1;

    il::Type type;
    fread(&type, sizeof(il::Type), 1, file);
    k += sizeof(il::Type);

    switch (type) {
      case il::Type::bool_t: {
        bool value = false;
        std::fread(&value, sizeof(bool), 1, file);
        k += sizeof(bool);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::integer_t: {
        const il::int_t value = read_varint(il::io, k, file);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::float_t: {
        float value = 0.0f;
        std::fread(&value, sizeof(float), 1, file);
        k += sizeof(float);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::double_t: {
        double value = 0.0;
        std::fread(&value, sizeof(double), 1, file);
        k += sizeof(double);
        config.set(std::move(string), il::Dynamic{value});
      } break;
      case il::Type::string_t: {
        il::int_t size = 0;
        std::fread(&size, sizeof(il::int_t), 1, file);
        k += sizeof(il::int_t);
        il::Array<char> raw_value{size + 1};
        std::fread(raw_value.data(), sizeof(char), size + 1, file);
        k += size + 1;
        il::String value = raw_value.data();
        config.set(std::move(string), il::Dynamic{std::move(value)});
      } break;
      case il::Type::array_of_double_t: {
        il::int_t size = 0;
        std::fread(&size, sizeof(il::int_t), 1, file);
        k += sizeof(il::int_t);
        il::Array<double> v{size};
        std::fread(v.data(), sizeof(double), size, file);
        k += sizeof(double) * size;
        config.set(std::move(string), il::Dynamic{std::move(v)});
      } break;
      case il::Type::array2d_of_uint8_t: {
        il::int_t size0 = 0;
        il::int_t size1 = 0;
        std::fread(&size0, sizeof(il::int_t), 1, file);
        std::fread(&size1, sizeof(il::int_t), 1, file);
        k += 2 * sizeof(il::int_t);
        il::Array2D<unsigned char> A{size0, size1};
        std::fread(A.data(), sizeof(double), size0 * size1, file);
        k += sizeof(unsigned char) * size0 * size1;
        config.set(std::move(string), il::Dynamic{std::move(A)});
      } break;
      case il::Type::array2d_of_double_t: {
        il::int_t size0 = 0;
        il::int_t size1 = 0;
        std::fread(&size0, sizeof(il::int_t), 1, file);
        std::fread(&size1, sizeof(il::int_t), 1, file);
        k += 2 * sizeof(il::int_t);
        il::Array2D<double> v{size0, size1};
        std::fread(v.data(), sizeof(double), size0 * size1, file);
        k += sizeof(double) * size0 * size1;
        config.set(std::move(string), il::Dynamic{std::move(v)});
      } break;
      case il::Type::hash_map_t:
      case il::Type::hash_map_array_t: {
        il::int_t n_map = 0;
        const il::int_t n = read_varint(il::io, n_map, file);
        il::int_t size = 0;
        il::int_t i = config.search(string);
        config.insert(std::move(string),
                      il::Dynamic{il::HashMapArray<il::String, il::Dynamic>{n}},
                      il::io, i);
        if (!config.found(i)) {
        }
        il::HashMapArray<il::String, il::Dynamic>& config_inner =
            config.value(i).as_hash_map_array();
        aux_load(size, il::io, config_inner, file);
        k += size;
      } break;
      default:
        IL_UNREACHABLE;
    }
  }
}

template <>
class LoadHelperData<il::HashMapArray<il::String, il::Dynamic>> {
 public:
  static il::HashMapArray<il::String, il::Dynamic> load(
      const il::String& filename, il::io_t, il::Status& status) {
    il::HashMapArray<il::String, il::Dynamic> ans{};

#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.as_c_string(), "rb");
#else
    il::UTF16String filename_utf16 = il::to_utf16(filename);
    std::FILE* file = _wfopen(filename_utf16.as_w_string(), L"rb");
#endif

    aux_load(-1, il::io, ans, file);

    const int error = std::fclose(file);
    if (error != 0) {
      status.set_error(il::Error::filesystem_cannot_close_file);
      return ans;
    }

    status.set_ok();
    return ans;
  }
};

template <>
class LoadHelperData<il::HashMap<il::String, il::Dynamic>> {
 public:
  static il::HashMap<il::String, il::Dynamic> load(const il::String& filename,
                                                   il::io_t,
                                                   il::Status& status) {
    il::HashMap<il::String, il::Dynamic> ans{};

#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.as_c_string(), "rb");
#else
    il::UTF16String filename_utf16 = il::to_utf16(filename);
    std::FILE* file = _wfopen(filename_utf16.as_w_string(), L"rb");
#endif

    aux_load(-1, il::io, ans, file);

    const int error = std::fclose(file);
    if (error != 0) {
      status.set_error(il::Error::filesystem_cannot_close_file);
      return ans;
    }

    status.set_ok();
    return ans;
  }
};

void aux_save(const il::HashMapArray<il::String, il::Dynamic>& data, il::io_t,
              il::int_t& n, std::FILE* file) {
  n = 0;

  for (il::int_t i = 0; i < data.size(); ++i) {
    il::String to_check = data.key(i).as_c_string();
    il::int_t my_size = data.key(i).size();
    IL_UNUSED(to_check);
    IL_UNUSED(my_size);

    const il::int_t string_size = data.key(i).size();
    write_varint(string_size + 1, il::io, n, file);
    std::fwrite(data.key(i).as_c_string(), sizeof(char), string_size, file);
    n += data.key(i).size() + 1;
    const il::Type type = data.value(i).type();
    std::fwrite(&type, sizeof(il::Type), 1, file);
    n += sizeof(il::Type);

    if (data.key(i) == il::String{"StatusParPhy2"}) {
      //      std::cout << "Hello world" << std::endl;
    }
    if (data.key(i) == il::String{"Delta"}) {
      //      std::cout << "Hello world" << std::endl;
    }

    switch (data.value(i).type()) {
      case il::Type::bool_t: {
        const bool value = data.value(i).to_bool();
        std::fwrite(&value, sizeof(bool), 1, file);
        n += sizeof(bool);
      } break;
      case il::Type::integer_t: {
        write_varint(data.value(i).to_integer(), il::io, n, file);
      } break;
      case il::Type::float_t: {
        const float value = data.value(i).to_float();
        std::fwrite(&value, sizeof(float), 1, file);
        n += sizeof(float);
      } break;
      case il::Type::double_t: {
        const double value = data.value(i).to_double();
        std::fwrite(&value, sizeof(double), 1, file);
        n += sizeof(double);
      } break;
      case il::Type::string_t: {
        const il::int_t size = data.value(i).as_string().size();
        std::fwrite(&size, sizeof(il::int_t), 1, file);
        std::fwrite(data.value(i).as_string().as_c_string(), sizeof(char),
                    size + 1, file);
        n += sizeof(il::int_t) + size + 1;
      } break;
      case il::Type::array_of_double_t: {
        const il::int_t size = data.value(i).as_array_of_double().size();
        std::fwrite(&size, sizeof(il::int_t), 1, file);
        std::fwrite(data.value(i).as_array_of_double().data(), sizeof(double),
                    size, file);
        n += sizeof(il::int_t) + sizeof(double) * size;
      } break;
      case il::Type::array2d_of_uint8_t: {
        const il::Array2D<unsigned char>& A =
            data.value(i).as_array2d_of_uint8();
        const il::int_t size0 = A.size(0);
        const il::int_t size1 = A.size(1);
        std::fwrite(&size0, sizeof(il::int_t), 1, file);
        std::fwrite(&size1, sizeof(il::int_t), 1, file);
        for (int j = 0; j < size1; ++j) {
          std::fwrite(A.data() + j * A.capacity(0), sizeof(double), size0,
                      file);
        }
        n += 2 * sizeof(il::int_t) + sizeof(unsigned char) * size0 * size1;
      } break;
      case il::Type::array2d_of_double_t: {
        const il::Array2D<double>& A = data.value(i).as_array2d_of_double();
        const il::int_t size0 = A.size(0);
        const il::int_t size1 = A.size(1);
        std::fwrite(&size0, sizeof(il::int_t), 1, file);
        std::fwrite(&size1, sizeof(il::int_t), 1, file);
        for (int j = 0; j < size1; ++j) {
          std::fwrite(A.data() + j * A.capacity(0), sizeof(double), size0,
                      file);
        }
        n += 2 * sizeof(il::int_t) + sizeof(double) * size0 * size1;
      } break;
      case il::Type::hash_map_array_t: {
        il::int_t n_map = 0;
        //        std::fwrite(&n_map, sizeof(il::int_t), 1, file);
        write_varint(data.value(i).as_hash_map_array().size(), il::io, n_map,
                     file);
        aux_save(data.value(i).as_hash_map_array(), il::io, n_map, file);
        //        std::fseek(file, -(n_map + sizeof(il::int_t)), SEEK_CUR);
        //        std::fwrite(&n_map, sizeof(il::int_t), 1, file);
        //        std::fseek(file, n_map, SEEK_CUR);
      } break;
      default:
        IL_UNREACHABLE;
    }
  }
  write_varint(0, il::io, n, file);
}

template <>
class SaveHelperData<il::HashMapArray<il::String, il::Dynamic>> {
 public:
  static void save(const il::HashMapArray<il::String, il::Dynamic>& data,
                   const il::String& filename, il::io_t, il::Status& status) {
#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.as_c_string(), "wb");
#else
    il::UTF16String filename_utf16 = il::to_utf16(filename);
    std::FILE* file = _wfopen(filename_utf16.as_w_string(), L"wb");
#endif

    il::int_t n = 0;
    aux_save(data, il::io, n, file);

    const int error = std::fclose(file);
    if (error != 0) {
      status.set_error(il::Error::filesystem_cannot_close_file);
      return;
    }

    status.set_ok();
    return;
  }
};

}  // namespace il

#endif  // IL_FILEPACK_H
