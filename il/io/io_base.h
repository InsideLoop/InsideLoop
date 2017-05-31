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
T load(const il::String& filename, il::io_t, il::Status& status) {
  return il::LoadHelper<T>::load(filename, il::io, status);
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
void save(const T& x, const il::String& filename, il::io_t,
          il::Status& status) {
  il::SaveHelper<T>::save(x, filename, il::io, status);
}

template <typename T>
void save(const T& x, const std::string& filename, il::io_t,
          il::Status& status) {
  il::String il_filename = filename.c_str();
  il::SaveHelper<T>::save(x, il_filename, il::io, status);
}

}  // namespace il

#endif  // IL_IO_BASE
