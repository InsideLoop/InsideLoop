//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_IO_BASE_H
#define IL_IO_BASE_H

#include <string>

#include <il/Status.h>
#include <il/String.h>

namespace il {

enum class FileType { kNpy, kToml, kData, kPng };

inline il::FileType fileType(const il::String& file, il::io_t,
                             il::Status& status) {
  if (file.hasSuffix(".npy")) {
    status.setOk();
    return il::FileType::kNpy;
  } else if (file.hasSuffix(".toml")) {
    status.setOk();
    return il::FileType::kToml;
  } else if (file.hasSuffix(".data")) {
    status.setOk();
    return il::FileType::kData;
  } else if (file.hasSuffix(".png")) {
    status.setOk();
    return il::FileType::kPng;
  } else {
    status.setError(il::Error::kUndefined);
    return il::FileType::kNpy;
  }
}

template <typename T>
class LoadHelper {
 public:
  static T load(const il::String& filename, il::io_t, il::Status& status);
};

template <typename T>
T LoadHelper<T>::load(const il::String& filename, il::io_t,
                      il::Status& status) {
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
  return T{};
}

template <typename T>
class LoadHelperToml {
 public:
  static T load(const il::String& filename, il::io_t, il::Status& status);
};

template <typename T>
T LoadHelperToml<T>::load(const il::String& filename, il::io_t,
                          il::Status& status) {
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
  return T{};
}

template <typename T>
class LoadHelperData {
 public:
  static T load(const il::String& filename, il::io_t, il::Status& status);
};

template <typename T>
T LoadHelperData<T>::load(const il::String& filename, il::io_t,
                          il::Status& status) {
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
  return T{};
}

template <typename T>
T load(const il::String& filename, il::io_t, il::Status& status) {
  const il::FileType ft = il::fileType(filename, il::io, status);
  if (!status.ok()) {
    status.rearm();
    return T{};
  }
  switch (ft) {
    case il::FileType::kToml:
      return il::LoadHelperToml<T>::load(filename, il::io, status);
      break;
    case il::FileType::kData:
      return il::LoadHelperData<T>::load(filename, il::io, status);
      break;
    default:
      return il::LoadHelper<T>::load(filename, il::io, status);
  }
}

template <typename T>
T load(const std::string& filename, il::io_t, il::Status& status) {
  il::String il_filename = filename.c_str();
  return il::LoadHelper<T>::load(il_filename, il::io, status);
}

template <typename T>
class SaveHelper {
 public:
  static void save(const T& x, const il::String& filename, il::io_t,
                   il::Status& status);
};

template <typename T>
void SaveHelper<T>::save(const T& x, const il::String& filename, il::io_t,
                         il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
class SaveHelperToml {
 public:
  static void save(const T& x, const il::String& filename, il::io_t,
                   il::Status& status);
};

template <typename T>
void SaveHelperToml<T>::save(const T& x, const il::String& filename, il::io_t,
                             il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
class SaveHelperData {
 public:
  static void save(const T& x, const il::String& filename, il::io_t,
                   il::Status& status);
};

template <typename T>
void SaveHelperData<T>::save(const T& x, const il::String& filename, il::io_t,
                             il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
class SaveHelperPng {
 public:
  static void save(const T& x, const il::String& filename, il::io_t,
                   il::Status& status);
};

template <typename T>
void SaveHelperPng<T>::save(const T& x, const il::String& filename, il::io_t,
                            il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.setError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
void save(const T& x, const il::String& filename, il::io_t,
          il::Status& status) {
  const il::FileType ft = il::fileType(filename, il::io, status);
  if (!status.ok()) {
    status.rearm();
    return;
  }
  switch (ft) {
    case il::FileType::kToml:
      il::SaveHelperToml<T>::save(x, filename, il::io, status);
      break;
    case il::FileType::kData:
      il::SaveHelperData<T>::save(x, filename, il::io, status);
      break;
    case il::FileType::kPng:
      il::SaveHelperPng<T>::save(x, filename, il::io, status);
      break;
    default:
      il::SaveHelper<T>::save(x, filename, il::io, status);
  }
}

}  // namespace il

#endif  // IL_IO_BASE
