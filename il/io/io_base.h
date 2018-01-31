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

#ifndef IL_IO_BASE_H
#define IL_IO_BASE_H

#include <string>

#include <il/Status.h>
#include <il/String.h>

namespace il {

enum class FileType { kNpy, kToml, kData, kPng };

inline il::FileType fileType(const il::String& file, il::io_t,
                             il::Status& status) {
  if (file.endsWith(".npy")) {
    status.SetOk();
    return il::FileType::kNpy;
  } else if (file.endsWith(".toml")) {
    status.SetOk();
    return il::FileType::kToml;
  } else if (file.endsWith(".data")) {
    status.SetOk();
    return il::FileType::kData;
  } else if (file.endsWith(".png")) {
    status.SetOk();
    return il::FileType::kPng;
  } else {
    status.SetError(il::Error::kUndefined);
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
  status.SetError(il::Error::kUnimplemented);
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
  status.SetError(il::Error::kUnimplemented);
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
  status.SetError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
  return T{};
}

template <typename T>
T load(const il::String& filename, il::io_t, il::Status& status) {
  const il::FileType ft = il::fileType(filename, il::io, status);
  if (!status.Ok()) {
    status.Rearm();
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
  il::String il_filename{il::StringType::kByte, filename.c_str(),
                         il::size(filename.c_str())};
  return il::LoadHelper<T>::load(il_filename, il::io, status);
}

template <typename T>
class SaveHelper {
 public:
  static void save(const T& x, const il::String& filename, il::io_t,
                   il::Status& status);
};

template <typename T, typename U>
class SaveHelperWithOptions {
 public:
  static void save(const T& x, const U& options, const il::String& filename,
                   il::io_t, il::Status& status);
};

template <typename T>
void SaveHelper<T>::save(const T& x, const il::String& filename, il::io_t,
                         il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.SetError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T, typename U>
void SaveHelperWithOptions<T, U>::save(const T& x, const U& options,
                                       const il::String& filename, il::io_t,
                                       il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  IL_UNUSED(options);
  status.SetError(il::Error::kUnimplemented);
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
  status.SetError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
class SaveHelperData {
 public:
  static void save(const T& x, const il::String& filename, il::io_t,
                   il::Status& status);
};

template <typename T, typename U>
class SaveHelperDataWithOptions {
 public:
  static void save(const T& x, const U& options, const il::String& filename,
                   il::io_t, il::Status& status);
};

template <typename T>
void SaveHelperData<T>::save(const T& x, const il::String& filename, il::io_t,
                             il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.SetError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T, typename U>
void SaveHelperDataWithOptions<T, U>::save(const T& x, const U&,
                                           const il::String& filename, il::io_t,
                                           il::Status& status) {
  IL_UNUSED(x);
  IL_UNUSED(filename);
  status.SetError(il::Error::kUnimplemented);
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
  status.SetError(il::Error::kUnimplemented);
  IL_SET_SOURCE(status);
}

template <typename T>
void save(const T& x, const il::String& filename, il::io_t,
          il::Status& status) {
  const il::FileType ft = il::fileType(filename, il::io, status);
  if (!status.Ok()) {
    status.Rearm();
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

template <typename T, typename U>
void save(const T& x, const U& option, const il::String& filename, il::io_t,
          il::Status& status) {
  const il::FileType ft = il::fileType(filename, il::io, status);
  if (!status.Ok()) {
    status.Rearm();
    return;
  }
  switch (ft) {
    case il::FileType::kToml:
      IL_UNREACHABLE;
      break;
    case il::FileType::kData:
      il::SaveHelperDataWithOptions<T, U>::save(x, option, filename, il::io,
                                                status);
      break;
    case il::FileType::kPng:
      IL_UNREACHABLE;
      break;
    default:
      IL_UNREACHABLE;
  }
}

}  // namespace il

#endif  // IL_IO_BASE
