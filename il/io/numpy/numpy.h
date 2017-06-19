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
#include <il/Status.h>
#include <il/String.h>
#include <il/io/io_base.h>

#ifdef IL_WINDOWS
#include <il/UTF16String.h>
#include <il/unicode.h>
#endif

namespace il {

template <typename T>
struct numpyType {
  static constexpr const char* value = "";
};

template <>
struct numpyType<int> {
  static constexpr const char* value = "i4";
};

template <>
struct numpyType<double> {
  static constexpr const char* value = "f8";
};

struct NumpyInfo {
  il::String type;
  il::Array<il::int_t> shape;
  bool fortran_order;
};

NumpyInfo getNumpyInfo(il::io_t, std::FILE* fp, il::Status& status);
void saveNumpyInfo(const NumpyInfo& numpy_info, il::io_t, std::FILE* fp,
                   il::Status& status);

template <typename T>
class SaveHelper<il::Array<T>> {
 public:
  static void save(const il::Array<T>& v, const il::String& filename, il::io_t,
                   il::Status& status) {
#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "wb");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"wb");
#endif
    if (!file) {
      status.setError(il::Error::kFilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return;
    }

    il::NumpyInfo numpy_info;
    numpy_info.shape = il::Array<il::int_t>{il::value, {v.size()}};
    numpy_info.type = il::numpyType<T>::value;
    numpy_info.fortran_order = true;

    il::Status info_status{};
    il::saveNumpyInfo(numpy_info, il::io, file, info_status);
    if (!info_status.ok()) {
      const int error = std::fclose(file);
      if (error != 0) {
        il::abort();
      }
      status = std::move(info_status);
      return;
    }

    std::size_t written = std::fwrite(v.data(), sizeof(T),
                                      static_cast<std::size_t>(v.size()), file);
    if (static_cast<il::int_t>(written) != v.size()) {
      status.setError(il::Error::kFilesystemCanNotWriteToFile);
      IL_SET_SOURCE(status);
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.setError(il::Error::kFilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return;
    }

    status.setOk();
    return;
  }
};

template <typename T>
class SaveHelper<il::Array2D<T>> {
 public:
  static void save(const il::Array2D<T>& A, const il::String& filename,
                   il::io_t, il::Status& status) {
#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "wb");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"wb");
#endif
    if (!file) {
      status.setError(il::Error::kFilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return;
    }

    il::NumpyInfo numpy_info;
    numpy_info.shape = il::Array<il::int_t>{il::value, {A.size(0), A.size(1)}};
    numpy_info.type = il::numpyType<T>::value;
    numpy_info.fortran_order = true;

    il::Status info_status{};
    il::saveNumpyInfo(numpy_info, il::io, file, info_status);
    if (!info_status.ok()) {
      const int error = std::fclose(file);
      if (error != 0) {
        il::abort();
      }
      status = std::move(info_status);
      return;
    }

    std::size_t written =
        std::fwrite(A.data(), sizeof(T),
                    static_cast<std::size_t>(A.size(0) * A.size(1)), file);
    if (static_cast<il::int_t>(written) != A.size(0) * A.size(1)) {
      status.setError(il::Error::kFilesystemCanNotWriteToFile);
      IL_SET_SOURCE(status);
      return;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.setError(il::Error::kFilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return;
    }

    status.setOk();
    return;
  }
};

template <typename T>
class LoadHelper<il::Array<T>> {
 public:
  static il::Array<T> load(const il::String& filename, il::io_t,
                           il::Status& status) {
    il::Array<T> v{};

#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "r+b");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"r+b");
#endif
    if (!file) {
      status.setError(il::Error::kFilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return v;
    }

    il::Status info_status{};
    il::NumpyInfo numpy_info = il::getNumpyInfo(il::io, file, info_status);
    if (!info_status.ok()) {
      status = std::move(info_status);
      return v;
    }

    if (!(numpy_info.type == il::numpyType<T>::value)) {
      status.setError(il::Error::kBinaryFileWrongType);
      IL_SET_SOURCE(status);
      return v;
    } else if (numpy_info.shape.size() != 1) {
      status.setError(il::Error::kBinaryFileWrongRank);
      IL_SET_SOURCE(status);
      return v;
    }

    v.resize(numpy_info.shape[0]);
    const std::size_t read = fread(v.data(), sizeof(T), v.size(), file);
    if (static_cast<il::int_t>(read) != v.size()) {
      status.setError(il::Error::kBinaryFileWrongFormat);
      IL_SET_SOURCE(status);
      return v;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.setError(il::Error::kFilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return v;
    }

    status.setOk();
    return v;
  }
};

template <typename T>
class LoadHelper<il::Array2D<T>> {
 public:
  static il::Array2D<T> load(const il::String& filename, il::io_t,
                             il::Status& status) {
    il::Array2D<T> v{};

#ifdef IL_UNIX
    std::FILE* file = std::fopen(filename.asCString(), "r+b");
#else
    il::UTF16String filename_utf16 = il::toUtf16(filename);
    std::FILE* file = _wfopen(filename_utf16.asWString(), L"r+b");
#endif
    if (!file) {
      status.setError(il::Error::kFilesystemFileNotFound);
      IL_SET_SOURCE(status);
      return v;
    }

    il::Status info_status{};
    il::NumpyInfo numpy_info = il::getNumpyInfo(il::io, file, info_status);
    if (!info_status.ok()) {
      status = std::move(info_status);
      return v;
    }

    if (!(numpy_info.type == il::numpyType<T>::value)) {
      status.setError(il::Error::kBinaryFileWrongType);
      IL_SET_SOURCE(status);
      return v;
    } else if (numpy_info.shape.size() != 2) {
      status.setError(il::Error::kBinaryFileWrongRank);
      IL_SET_SOURCE(status);
      return v;
    } else if (!numpy_info.fortran_order) {
      status.setError(il::Error::kBinaryFileWrongEndianness);
      IL_SET_SOURCE(status);
      return v;
    }

    v.resize(numpy_info.shape[0], numpy_info.shape[1]);
    const il::int_t n = v.size(0) * v.size(1);
    const std::size_t read = fread(v.data(), sizeof(T), n, file);
    if (static_cast<il::int_t>(read) != n) {
      status.setError(il::Error::kBinaryFileWrongFormat);
      IL_SET_SOURCE(status);
      return v;
    }

    const int error = std::fclose(file);
    if (error != 0) {
      status.setError(il::Error::kFilesystemCanNotCloseFile);
      IL_SET_SOURCE(status);
      return v;
    }

    status.setOk();
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
    local_status.abortOnError();

    il::String filename_column = filename;
    filename_column.append(".column");
    auto column =
        il::load<il::Array<il::int_t>>(filename_column, il::io, local_status);
    local_status.abortOnError();

    il::String filename_element = filename;
    filename_element.append(".element");
    auto element =
        il::load<il::Array<double>>(filename_element, il::io, local_status);
    local_status.abortOnError();

    const il::int_t n = row.size() - 1;
    status.setOk();
    return il::SparseMatrixCSR<il::int_t, double>{
        n, n, std::move(column), std::move(row), std::move(element)};
  }
};
}  // namespace il

#endif  // IL_NUMPY_H
