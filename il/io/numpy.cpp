//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/io/numpy.h>

#include <il/String.h>
#include <il/StringView.h>
#include <il/container/string/algorithm_string.h>

namespace il {

NumpyInfo get_numpy_info(il::io_t, std::FILE* fp, il::Status& status) {
  NumpyInfo numpy_info;

  char first_buffer[10];
  StringView buffer{first_buffer, 10};

  // Read the first 10 bytes of the files. It should contain:
  // - The magic string "\x93NUMPY"
  // - The major version number
  // - The minor version number
  // - The number of bytes for the header length
  //
  std::size_t count = 10;
  const std::size_t read = fread(buffer.begin(), sizeof(char), count, fp);
  if (read != count) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }
  if (!(buffer.substring(0, 6) == "\x93NUMPY")) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }
  unsigned char major_version = buffer[6];
  unsigned char minor_version = buffer[7];
  unsigned short header_length =
      *reinterpret_cast<unsigned short*>(buffer.begin() + 8);
  if (major_version != 1 || minor_version != 0) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }

  // Read the header
  //
  il::Array<char> second_buffer{header_length + 1};
  StringView header = StringView{second_buffer.begin(), header_length + 1};
  char* success = fgets(header.begin(), header_length + 1, fp);
  if (success == nullptr || header[header.size() - 2] != '\n') {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }

  // Read the endianness, the type of the Numpy array, and the byte size
  //
  const il::int_t i4 = il::search("descr", header);
  if (i4 == -1 || i4 + 12 >= header.size()) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }
  const bool little_endian = header[i4 + 9] == '<' || header[i4 + 9] == '|';
  if (!little_endian) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }

  ConstStringView type_string = header.substring(i4 + 10);
  const il::int_t i5 = il::search("'", type_string);
  numpy_info.type = il::String{type_string.begin(), i5};

  // Read the ordering for multidimensional arrays
  //
  const il::int_t i0 = il::search("fortran_order", header);
  if (i0 == -1 || i0 + 20 > header.size()) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }

  ConstStringView fortran_order_string = header.substring(i0 + 16, i0 + 20);
  numpy_info.fortran_order = (fortran_order_string == "True") ? true : false;

  // Read the dimensions
  //
  const il::int_t i1 = il::search("(", header);
  const il::int_t i2 = il::search(")", header);
  if (i1 == -1 || i2 == -1 || i2 - i1 <= 1) {
    status.set_error(il::ErrorCode::wrong_file_format);
    return numpy_info;
  }
  ConstStringView shape_string = header.substring(i1 + 1, i2);
  if (shape_string.back() == ',') {
    numpy_info.shape.resize(1);
  } else {
    const il::int_t n_dim = il::count(',', shape_string) + 1;
    numpy_info.shape.resize(n_dim);
  }
  for (il::int_t i = 0; i < numpy_info.shape.size(); ++i) {
    const il::int_t i3 = il::search(",", shape_string);
    numpy_info.shape[i] = std::atoll(shape_string.substring(0, i3).c_string());
    shape_string = shape_string.substring(i3 + 1);
  }

  status.set_ok();
  return numpy_info;
}

void save_numpy_info(const NumpyInfo& numpy_info, il::io_t, std::FILE* fp,
                     il::Status& status) {
  il::String header{};
  header.append("{'descr': '");
  // Little endian
  header.append("<");
  // type of the array
  header.append(numpy_info.type);
  // ordering
  header.append("', 'fortran_order': ");
  header.append(numpy_info.fortran_order ? "True" : "False");
  // shape of the array
  // The buffer can hold enough digits for any 64-bit integer
  char buffer[21];
  header.append(", 'shape': (");
  std::sprintf(buffer, "%td", numpy_info.shape[0]);
  header.append(buffer);
  for (il::int_t i = 1; i < numpy_info.shape.size(); ++i) {
    header.append(", ");
    std::sprintf(buffer, "%td", numpy_info.shape[i]);
    header.append(buffer);
  }
  if (numpy_info.shape.size() == 1) {
    header.append(",");
  }
  header.append("), }");
  il::int_t remainder = 16 - (10 + header.size()) % 16;
  header.append(remainder, ' ');
  header.back() = '\n';

  il::String magic{};
  magic.append("\x93NUMPY");
  // Numpy format major version
  magic.append(static_cast<char>(0x01));
  // Numpy format minor version
  magic.append(static_cast<char>(0x00));
  // Size of the header
  il::String short_int = "  ";
  *(reinterpret_cast<unsigned short*>(short_int.begin())) =
      static_cast<unsigned short>(header.size());
  magic.append(short_int);
  magic.append(header);

  std::size_t written = std::fwrite(magic.c_string(), sizeof(char),
                                    static_cast<std::size_t>(magic.size()), fp);
  if (static_cast<il::int_t>(written) != magic.size()) {
    status.set_error(il::ErrorCode::cannot_write_to_file);
    return;
  }

  status.set_ok();
}
}
