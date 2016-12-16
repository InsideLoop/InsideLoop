//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_NUMPY_H
#define IL_NUMPY_H

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/core/Status.h>

#include <il/SparseArray2C.h>
#include <zlib.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

struct NpyArray {
  char* data;
  std::vector<unsigned int> shape;
  unsigned int word_size;
  bool fortran_order;
  void destruct() { delete[] data; }
};

struct npz_t : public std::map<std::string, NpyArray> {
  void destruct() {
    npz_t::iterator it = this->begin();
    for (; it != this->end(); ++it) (*it).second.destruct();
  }
};

//char BigEndianTest();
char map_type(const std::type_info& t);
template <typename T>
std::vector<char> create_npy_header(const T* data, const unsigned int* shape,
                                    const unsigned int ndims,
                                    bool fortran_order);
void parse_npy_header(FILE* fp, unsigned int& word_size, unsigned int*& shape,
                      unsigned int& ndims, bool& fortran_order, char* type);
void parse_zip_footer(FILE* fp, unsigned short& nrecs,
                      unsigned int& global_header_size,
                      unsigned int& global_header_offset);
npz_t npz_load(std::string fname);
NpyArray npz_load(std::string fname, std::string varname);
NpyArray npy_load(std::string fname);

template <typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
  // write in little endian
  for (char byte = 0; byte < static_cast<char>(sizeof(T)); byte++) {
    char val = *((char*)&rhs + byte);
    lhs.push_back(val);
  }
  return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);

template <typename T>
std::string tostring(T i, int pad = 0, char padval = ' ') {
  (void)pad;
  (void)padval;
  std::stringstream s;
  s << i;
  return s.str();
}

template <typename T>
void npy_save(std::string fname, const T* data, const unsigned int* shape,
              const unsigned int ndims, bool fortran_order, std::string mode,
              il::io_t, il::Status &status) {
  FILE* fp = NULL;

  if (mode == "a") fp = fopen(fname.c_str(), "r+b");

  if (fp) {
    // file exists. we need to append to it. read the header, modify the array
    // size
    unsigned int word_size, tmp_dims;
    unsigned int* tmp_shape = 0;
    bool fortran_order;
    char type;
    parse_npy_header(fp, word_size, tmp_shape, tmp_dims, fortran_order, &type);
    assert(!fortran_order);

    if (word_size != sizeof(T)) {
      std::cout << "libnpy error: " << fname << " has word size " << word_size
                << " but npy_save appending data sized " << sizeof(T) << "\n";
      assert(word_size == sizeof(T));
    }
    if (tmp_dims != ndims) {
      std::cout << "libnpy error: npy_save attempting to append misdimensioned "
                   "data to "
                << fname << "\n";
      assert(tmp_dims == ndims);
    }

    for (int i = 1; i < static_cast<int>(ndims); i++) {
      if (shape[i] != tmp_shape[i]) {
        std::cout
            << "libnpy error: npy_save attempting to append misshaped data to "
            << fname << "\n";
        assert(shape[i] == tmp_shape[i]);
      }
    }
    tmp_shape[0] += shape[0];

    fseek(fp, 0, SEEK_SET);
    std::vector<char> header = create_npy_header(data, tmp_shape, ndims, false);
    fwrite(&header[0], sizeof(char), header.size(), fp);
    fseek(fp, 0, SEEK_END);

    delete[] tmp_shape;
  } else {
    fp = fopen(fname.c_str(), "wb");
    if (fp) {
      std::vector<char> header =
          create_npy_header(data, shape, ndims, fortran_order);
      fwrite(&header[0], sizeof(char), header.size(), fp);
    }
  }

  if (fp) {
    unsigned int nels = 1;
    for (int i = 0; i < static_cast<int>(ndims); i++) {
      nels *= shape[i];
    }
    fwrite(data, sizeof(T), nels, fp);
    fclose(fp);
    status.set(il::ErrorCode::ok);
  } else {
    status.set(il::ErrorCode::not_found);
  }
}

template <typename T>
void npz_save(std::string zipname, std::string fname, const T* data,
              const unsigned int* shape, const unsigned int ndims,
              std::string mode = "w") {
  // first, append a .npy to the fname
  fname += ".npy";

  // now, on with the show
  FILE* fp = NULL;
  unsigned short nrecs = 0;
  unsigned int global_header_offset = 0;
  std::vector<char> global_header;

  if (mode == "a") fp = fopen(zipname.c_str(), "r+b");

  if (fp) {
    // zip file exists. we need to add a new npy file to it.
    // first read the footer. this gives us the offset and size of the global
    // header
    // then read and store the global header.
    // below, we will write the the new data at the start of the global header
    // then append the global header and footer below it
    unsigned int global_header_size;
    parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);
    fseek(fp, global_header_offset, SEEK_SET);
    global_header.resize(global_header_size);
    size_t res = fread(&global_header[0], sizeof(char), global_header_size, fp);
    if (res != global_header_size) {
      throw std::runtime_error(
          "npz_save: header read error while adding to existing zip");
    }
    fseek(fp, global_header_offset, SEEK_SET);
  } else {
    fp = fopen(zipname.c_str(), "wb");
  }

  std::vector<char> npy_header = create_npy_header(data, shape, ndims);

  unsigned long nels = 1;
  for (int m = 0; m < static_cast<int>(ndims); m++) nels *= shape[m];
  int nbytes = nels * sizeof(T) + npy_header.size();

  // get the CRC of the data to be added
  unsigned int crc =
      crc32(0L, (unsigned char*)&npy_header[0], npy_header.size());
  crc = crc32(crc, (unsigned char*)data, nels * sizeof(T));

  // build the local header
  std::vector<char> local_header;
  local_header += "PK";                          // first part of sig
  local_header += (unsigned short)0x0403;        // second part of sig
  local_header += (unsigned short)20;            // min version to extract
  local_header += (unsigned short)0;             // general purpose bit flag
  local_header += (unsigned short)0;             // compression method
  local_header += (unsigned short)0;             // file last mod time
  local_header += (unsigned short)0;             // file last mod date
  local_header += (unsigned int)crc;             // crc
  local_header += (unsigned int)nbytes;          // compressed size
  local_header += (unsigned int)nbytes;          // uncompressed size
  local_header += (unsigned short)fname.size();  // fname length
  local_header += (unsigned short)0;             // extra field length
  local_header += fname;

  // build global header
  global_header += "PK";                    // first part of sig
  global_header += (unsigned short)0x0201;  // second part of sig
  global_header += (unsigned short)20;      // version made by
  global_header.insert(global_header.end(), local_header.begin() + 4,
                       local_header.begin() + 30);
  global_header += (unsigned short)0;  // file comment length
  global_header += (unsigned short)0;  // disk number where file starts
  global_header += (unsigned short)0;  // internal file attributes
  global_header += (unsigned int)0;    // external file attributes
  global_header +=
      (unsigned int)global_header_offset;  // relative offset of local file
                                           // header, since it begins where the
                                           // global header used to begin
  global_header += fname;

  // build footer
  std::vector<char> footer;
  footer += "PK";                         // first part of sig
  footer += (unsigned short)0x0605;       // second part of sig
  footer += (unsigned short)0;            // number of this disk
  footer += (unsigned short)0;            // disk where footer starts
  footer += (unsigned short)(nrecs + 1);  // number of records on this disk
  footer += (unsigned short)(nrecs + 1);  // total number of records
  footer += (unsigned int)global_header.size();  // nbytes of global headers
  footer += (unsigned int)(global_header_offset + nbytes +
                           local_header.size());  // offset of start of global
                                                  // headers, since global
                                                  // header now starts after
                                                  // newly written array
  footer += (unsigned short)0;                    // zip file comment length

  // write everything
  fwrite(&local_header[0], sizeof(char), local_header.size(), fp);
  fwrite(&npy_header[0], sizeof(char), npy_header.size(), fp);
  fwrite(data, sizeof(T), nels, fp);
  fwrite(&global_header[0], sizeof(char), global_header.size(), fp);
  fwrite(&footer[0], sizeof(char), footer.size(), fp);
  fclose(fp);
}

template <typename T>
std::vector<char> create_npy_header(const T* data, const unsigned int* shape,
                                    const unsigned int ndims,
                                    bool fortran_order) {
  (void)data;
  std::vector<char> dict;
  dict += "{'descr': '";
  dict += '<'; //BigEndianTest();
  dict += map_type(typeid(T));
  dict += tostring(sizeof(T));
  dict += "', 'fortran_order': ";
  dict += fortran_order ? "True" : "False";
  dict += ", 'shape': (";
  dict += tostring(shape[0]);
  for (int i = 1; i < static_cast<int>(ndims); i++) {
    dict += ", ";
    dict += tostring(shape[i]);
  }
  if (ndims == 1) dict += ",";
  dict += "), }";
  // pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10
  // bytes. dict needs to end with \n
  int remainder = 16 - (10 + dict.size()) % 16;
  dict.insert(dict.end(), remainder, ' ');
  dict.back() = '\n';

  std::vector<char> header;
  header += (char)0x93;
  header += "NUMPY";
  header += (char)0x01;  // major version of numpy format
  header += (char)0x00;  // minor version of numpy format
  header += (unsigned short)dict.size();
  header.insert(header.end(), dict.begin(), dict.end());

  return header;
}
}

namespace il {

template <typename T>
void save(const il::Array<T>& v, const std::string& filename, il::io_t,
          il::Status &status) {
  il::Array<unsigned int> shape{il::value,
                                {static_cast<unsigned int>(v.size())}};
  bool fortran_order = false;
  cnpy::npy_save(filename, v.data(), shape.data(), shape.size(), fortran_order,
                 "w", il::io, status);
}

template <typename T>
void save(const il::Array2D<T>& v, const std::string& filename, il::io_t,
          il::Status &status) {
  il::Array<unsigned int> shape{il::value,
                                {static_cast<unsigned int>(v.size(0)),
                                 static_cast<unsigned int>(v.size(1))}};
  bool fortran_order = true;
  cnpy::npy_save(filename, v.data(), shape.data(), shape.size(), fortran_order,
                 "w", il::io, status);
}

inline void save(const il::SparseArray2C<double>& A,
                 const std::string& filename, il::io_t, il::Status &status) {
  bool fortran_order = false;
  il::Array<unsigned int> shape_column{
      il::value, {static_cast<unsigned int>(A.nb_nonzeros())}};
  cnpy::npy_save(filename + std::string{".column"}, A.column_data(),
                 shape_column.data(), shape_column.size(), fortran_order, "w",
                 il::io, status);

  il::Array<unsigned int> shape_row{il::value,
                                    {static_cast<unsigned int>(A.size(0) + 1)}};
  cnpy::npy_save(filename + std::string{".row"}, A.row_data(), shape_row.data(),
                 shape_row.size(), fortran_order, "w", il::io, status);

  il::Array<unsigned int> shape_element{
      il::value, {static_cast<unsigned int>(A.nb_nonzeros())}};
  cnpy::npy_save(filename + std::string{".element"}, A.element_data(),
                 shape_element.data(), shape_element.size(), fortran_order, "w",
                 il::io, status);
}

template <typename T>
T load(const std::string& filename, il::io_t, il::Status &status) {
  (void)filename;
  status.set(il::ErrorCode::unimplemented);
  return T{};
}

template <>
inline il::Array<int> load(const std::string& filename, il::io_t,
                           il::Status &status) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    status.set(il::ErrorCode::not_found);
    return il::Array<int>{};
  }

  unsigned int* shape;
  unsigned int ndims, word_size;
  bool fortran_order;
  char type;
  cnpy::parse_npy_header(fp, word_size, shape, ndims, fortran_order, &type);

  unsigned long long size =
      1;  // long long so no overflow when multiplying by word_size
  for (unsigned int i = 0; i < ndims; ++i) {
    size *= shape[i];
  }

  IL_ASSERT(ndims == 1);
  IL_ASSERT(word_size == sizeof(int));

  il::Array<int> v{static_cast<il::int_t>(size)};
  std::size_t nread = fread(v.data(), word_size, size, fp);
  if (nread != size) {
    status.set(il::ErrorCode::wrong_file_format);
    return v;
  }

  fclose(fp);

  status.set(il::ErrorCode::ok);
  return v;
}

template <>
inline il::Array<double> load(const std::string& filename, il::io_t,
                              il::Status &status) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    status.set(il::ErrorCode::not_found);
    return il::Array<double>{};
  }

  unsigned int* shape;
  unsigned int ndims, word_size;
  bool fortran_order;
  char type;
  cnpy::parse_npy_header(fp, word_size, shape, ndims, fortran_order, &type);

  unsigned long long size =
      1;  // long long so no overflow when multiplying by word_size
  for (unsigned int i = 0; i < ndims; ++i) {
    size *= shape[i];
  }

  IL_ASSERT(ndims == 1);
  IL_ASSERT(word_size == sizeof(double));

  il::Array<double> v{static_cast<il::int_t>(size)};
  std::size_t nread = fread(v.data(), word_size, size, fp);
  if (nread != size) {
    status.set(il::ErrorCode::wrong_file_format);
    return v;
  }

  fclose(fp);

  status.set(il::ErrorCode::ok);
  return v;
}

template <>
inline il::Array2D<int> load(const std::string& filename, il::io_t,
                             il::Status &status) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    status.set(il::ErrorCode::not_found);
    return il::Array2D<int>{};
  }

  unsigned int* shape;
  unsigned int ndims, word_size;
  bool fortran_order;
  char type;
  cnpy::parse_npy_header(fp, word_size, shape, ndims, fortran_order, &type);
  if (type != cnpy::map_type(typeid(int))) {
    status.set(il::ErrorCode::wrong_type);
    return il::Array2D<int>{};
  }

  unsigned long long size =
      1;  // long long so no overflow when multiplying by word_size
  for (unsigned int i = 0; i < ndims; ++i) {
    size *= shape[i];
  }

  IL_ASSERT(ndims == 2);
  IL_ASSERT(word_size == sizeof(int));
  IL_ASSERT(fortran_order);

  il::Array2D<int> A{static_cast<il::int_t>(shape[0]),
                     static_cast<il::int_t>(shape[1])};

  std::size_t nread = fread(A.data(), word_size, size, fp);
  if (nread != size) {
    status.set(il::ErrorCode::wrong_file_format);
    return A;
  }

  fclose(fp);
  status.set(il::ErrorCode::ok);
  return A;
}

template <>
inline il::Array2D<double> load(const std::string& filename, il::io_t,
                                il::Status &status) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    status.set(il::ErrorCode::not_found);
    return il::Array2D<double>{};
  }

  unsigned int* shape;
  unsigned int ndims, word_size;
  bool fortran_order;
  char type;
  cnpy::parse_npy_header(fp, word_size, shape, ndims, fortran_order, &type);
  if (type != cnpy::map_type(typeid(double))) {
    status.set(il::ErrorCode::wrong_type);
    return il::Array2D<double>{};
  }

  unsigned long long size =
      1;  // long long so no overflow when multiplying by word_size
  for (unsigned int i = 0; i < ndims; ++i) {
    size *= shape[i];
  }

  IL_ASSERT(ndims == 2);
  IL_ASSERT(word_size == sizeof(double));
  IL_ASSERT(fortran_order);

  il::Array2D<double> A{static_cast<il::int_t>(shape[0]),
                        static_cast<il::int_t>(shape[1])};

  std::size_t nread = fread(A.data(), word_size, size, fp);
  if (nread != size) {
    status.set(il::ErrorCode::wrong_file_format);
    return A;
  }

  fclose(fp);
  status.set(il::ErrorCode::ok);
  return A;
}

template <>
inline il::SparseArray2C<double> load(const std::string& filename, il::io_t,
                                      il::Status &status) {
  il::Status local_status{};
  auto row =
      il::load<il::Array<il::int_t>>(filename + std::string{".row"}, il::io, local_status);
  local_status.abort_on_error();
  auto column = il::load<il::Array<il::int_t>>(filename + std::string{".column"},
                                         il::io, local_status);
  local_status.abort_on_error();
  auto element = il::load<il::Array<double>>(filename + std::string{".element"},
                                             il::io, local_status);
  local_status.abort_on_error();

  const il::int_t n = row.size() - 1;
  status.set(il::ErrorCode::ok);
  return il::SparseArray2C<double>{n, n, std::move(column), std::move(row),
                                   std::move(element)};
}
}
#endif  // IL_NUMPY_H
