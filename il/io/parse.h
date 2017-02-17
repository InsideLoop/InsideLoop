
//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstring>

#include <il/core/Status.h>

namespace il {

template <typename T>
T parse(const std::string& src, il::io_t, il::Status& status) {
  // unimplemented
  return T{};
}

template <>
int parse<int>(const std::string& src, il::io_t, il::Status& status) {
  char* end = nullptr;
  const int base = 10;
  const long int long_value = strtol(src.c_str(), &end, base);

  if (*end != '\0') {
    status.set_error(il::ErrorCode::wrong_input);
    return 0;
  }

  const int value = static_cast<int>(long_value);
  if (long_value == std::numeric_limits<long>::min() ||
      long_value == std::numeric_limits<long>::max() || value != long_value) {
    status.set_error(il::ErrorCode::wrong_input);
    return 0;
  }

  status.set_error(il::ErrorCode::ok);
  return value;
}

template <>
double parse<double>(const std::string& src, il::io_t, il::Status& status) {
  char* end = nullptr;
  const double value = strtod(src.c_str(), &end);

  if (*end != '\0') {
    status.set_error(il::ErrorCode::wrong_input);
    return 0;
  }

  status.set_error(il::ErrorCode::ok);
  return value;
}


}
