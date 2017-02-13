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

#include <string>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/SparseMatrixCSR.h>
#include <il/String.h>
#include <il/core/Status.h>
#include <il/io/io_base.h>

namespace il {

template <typename T>
struct numpy_type {
  static constexpr const char* value = "";
};

template <>
struct numpy_type<int> {
  static constexpr const char* value = "i4";
};

template <>
struct numpy_type<double> {
  static constexpr const char* value = "f8";
};

struct NumpyInfo {
  il::String type;
  il::Array<il::int_t> shape;
  bool fortran_order;
};

NumpyInfo get_numpy_info(il::io_t, std::FILE* fp, il::Status& status);
void save_numpy_info(const NumpyInfo& numpy_info, il::io_t, std::FILE* fp,
                     il::Status& status);

template <typename T>
class SaveHelper<il::Array<T>> {
 public:
  static void save(const il::Array<T>& v, const il::String& filename, il::io_t,
                   il::Status& status) {
    std::FILE* file = std::fopen(filename.c_string(), "wb");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return;
    }

    il::NumpyInfo numpy_info;
    numpy_info.shape = il::Array<il::int_t>{il::value, {v.size()}};
    numpy_info.type = il::numpy_type<T>::value;
    numpy_info.fortran_order = true;

    il::Status info_status{};
    il::save_numpy_info(numpy_info, il::io, file, info_status);
    if (!info_status.ok()) {
      const int error = std::fclose(file);
      if (error != 0) {
        il::abort();
      }
      status.set(info_status.error_code());
      return;
    }

    std::size_t written = std::fwrite(v.data(), sizeof(T),
                                      static_cast<std::size_t>(v.size()), file);
    if (static_cast<il::int_t>(written) != v.size()) {
      status.set(il::ErrorCode::cannot_write_to_file);
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return;
    }

    status.set(il::ErrorCode::ok);
    return;
  }
};

template <typename T>
class SaveHelper<il::Array2D<T>> {
 public:
  static void save(const il::Array2D<T>& A, const il::String& filename,
                   il::io_t, il::Status& status) {
    std::FILE* file = std::fopen(filename.c_string(), "wb");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return;
    }

    il::NumpyInfo numpy_info;
    numpy_info.shape = il::Array<il::int_t>{il::value, {A.size(0), A.size(1)}};
    numpy_info.type = il::numpy_type<T>::value;
    numpy_info.fortran_order = true;

    il::Status info_status{};
    il::save_numpy_info(numpy_info, il::io, file, info_status);
    if (!info_status.ok()) {
      const int error = std::fclose(file);
      if (error != 0) {
        il::abort();
      }
      status.set(info_status.error_code());
      return;
    }

    std::size_t written =
        std::fwrite(A.data(), sizeof(T),
                    static_cast<std::size_t>(A.size(0) * A.size(1)), file);
    if (static_cast<il::int_t>(written) != A.size(0) * A.size(1)) {
      status.set(il::ErrorCode::cannot_write_to_file);
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return;
    }

    status.set(il::ErrorCode::ok);
    return;
  }
};


template <typename T>
class LoadHelper<il::Array<T>> {
 public:
  static il::Array<T> load(const il::String& filename, il::io_t,
                           il::Status& status) {
    il::Array<T> v{};

    std::FILE* file = std::fopen(filename.c_string(), "r+b");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return v;
    }

    il::Status info_status{};
    il::NumpyInfo numpy_info = il::get_numpy_info(il::io, file, info_status);
    if (!info_status.ok()) {
      status.set(info_status.error_code());
      return v;
    }

    if (!(numpy_info.type == il::numpy_type<T>::value)) {
      status.set(il::ErrorCode::wrong_type);
      return v;
    } else if (numpy_info.shape.size() != 1) {
      status.set(il::ErrorCode::wrong_rank);
      return v;
    }

    v.resize(numpy_info.shape[0]);
    const std::size_t read = fread(v.data(), sizeof(T), v.size(), file);
    if (static_cast<il::int_t>(read) != v.size()) {
      status.set(il::ErrorCode::wrong_file_format);
      return v;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return v;
    }

    status.set(il::ErrorCode::ok);
    return v;
  }
};

template <typename T>
class LoadHelper<il::Array2D<T>> {
 public:
  static il::Array2D<T> load(const il::String& filename, il::io_t,
                             il::Status& status) {
    il::Array2D<T> v{};

    std::FILE* file = std::fopen(filename.c_string(), "r+b");
    if (!file) {
      status.set(il::ErrorCode::file_not_found);
      return v;
    }

    il::Status info_status{};
    il::NumpyInfo numpy_info = il::get_numpy_info(il::io, file, info_status);
    if (!info_status.ok()) {
      status.set(info_status.error_code());
      return v;
    }

    if (!(numpy_info.type == il::numpy_type<T>::value)) {
      status.set(il::ErrorCode::wrong_type);
      return v;
    } else if (numpy_info.shape.size() != 2) {
      status.set(il::ErrorCode::wrong_rank);
      return v;
    } else if (!numpy_info.fortran_order) {
      status.set(il::ErrorCode::wrong_order);
      return v;
    }

    v.resize(numpy_info.shape[0], numpy_info.shape[1]);
    const il::int_t n = v.size(0) * v.size(1);
    const std::size_t read = fread(v.data(), sizeof(T), n, file);
    if (static_cast<il::int_t>(read) != n) {
      status.set(il::ErrorCode::wrong_file_format);
      return v;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.set(il::ErrorCode::cannot_close_file);
      return v;
    }

    status.set(il::ErrorCode::ok);
    return v;
  }
};

template <>
class LoadHelper<il::SparseMatrixCSR<il::int_t, double>> {
 public:
  static il::SparseMatrixCSR<il::int_t, double> load(const il::String& filename,
                                                     il::io_t,
                                                     il::Status& status) {
    il::Status local_status{};
    il::String filename_row = filename;
    filename_row.append(".row");
    auto row =
        il::load<il::Array<il::int_t>>(filename_row, il::io, local_status);
    local_status.abort_on_error();

    il::String filename_column = filename;
    filename_column.append(".column");
    auto column =
        il::load<il::Array<il::int_t>>(filename_column, il::io, local_status);
    local_status.abort_on_error();

    il::String filename_element = filename;
    filename_element.append(".element");
    auto element =
        il::load<il::Array<double>>(filename_element, il::io, local_status);
    local_status.abort_on_error();

    const il::int_t n = row.size() - 1;
    status.set(il::ErrorCode::ok);
    return il::SparseMatrixCSR<il::int_t, double>{
        n, n, std::move(column), std::move(row), std::move(element)};
  }
};
}

#endif  // IL_NUMPY_H
