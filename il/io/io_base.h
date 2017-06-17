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

enum class FileType { npy, toml, data, png };

inline il::FileType file_type(const il::String& file, il::io_t,
                              il::Status& status) {
  if (file.has_suffix(".npy")) {
    status.set_ok();
    return il::FileType::npy;
  } else if (file.has_suffix(".toml")) {
    status.set_ok();
    return il::FileType::toml;
  } else if (file.has_suffix(".data")) {
    status.set_ok();
    return il::FileType::data;
  } else if (file.has_suffix(".png")) {
    status.set_ok();
    return il::FileType::png;
  } else {
    status.set_error(il::Error::undefined);
    return il::FileType::npy;
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
  status.set_error(il::Error::unimplemented);
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
  status.set_error(il::Error::unimplemented);
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
  status.set_error(il::Error::unimplemented);
  IL_SET_SOURCE(status);
  return T{};
}

template <typename T>
T load(const il::String& filename, il::io_t, il::Status& status) {
  const il::FileType ft = il::file_type(filename, il::io, status);
  if (!status.ok()) {
    status.rearm();
    return T{};
  }
  switch (ft) {
    case il::FileType::toml:
      return il::LoadHelperToml<T>::load(filename, il::io, status);
      break;
    case il::FileType::data:
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
  status.set_error(il::Error::unimplemented);
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
  status.set_error(il::Error::unimplemented);
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
  status.set_error(il::Error::unimplemented);
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
  status.set_error(il::Error::unimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
void save(const T& x, const il::String& filename, il::io_t,
          il::Status& status) {
  const il::FileType ft = il::file_type(filename, il::io, status);
  if (!status.ok()) {
    status.rearm();
    return;
  }
  switch (ft) {
    case il::FileType::toml:
      il::SaveHelperToml<T>::save(x, filename, il::io, status);
      break;
    case il::FileType::data:
      il::SaveHelperData<T>::save(x, filename, il::io, status);
      break;
    case il::FileType::png:
      il::SaveHelperPng<T>::save(x, filename, il::io, status);
      break;
    default:
      il::SaveHelper<T>::save(x, filename, il::io, status);
  }
}


}  // namespace il

#endif  // IL_IO_BASE
