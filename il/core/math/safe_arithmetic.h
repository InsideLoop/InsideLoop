//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SAFE_ARITHMETIC_H
#define IL_SAFE_ARITHMETIC_H

#include <limits>

#include <il/base.h>
#include <il/core/Status.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// 32-bit integer
////////////////////////////////////////////////////////////////////////////////

#ifdef INT64_MAX
inline std::int32_t safe_sum(std::int32_t a, std::int32_t b, il::io_t,
                             bool& error) {
  const std::int64_t a_64 = a;
  const std::int64_t b_64 = b;
  const std::int64_t sum_64 = a_64 + b_64;
  if (sum_64 > std::numeric_limits<std::int32_t>::max()) {
    error = true;
    return 0;
  } else if (sum_64 < std::numeric_limits<std::int32_t>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<std::int32_t>(sum_64);
  }
}
#endif

#ifdef INT64_MAX
inline std::int32_t safe_difference(std::int32_t a, std::int32_t b, il::io_t,
                                    bool& error) {
  const std::int64_t a_64 = a;
  const std::int64_t b_64 = b;
  const std::int64_t difference_64 = a_64 - b_64;
  if (difference_64 > std::numeric_limits<std::int32_t>::max()) {
    error = true;
    return 0;
  } else if (difference_64 < std::numeric_limits<std::int32_t>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<std::int32_t>(difference_64);
  }
}
#endif

#ifdef INT64_MAX
inline std::int32_t safe_product(std::int32_t a, std::int32_t b, il::io_t,
                                 bool& error) {
  const std::int64_t a_64 = a;
  const std::int64_t b_64 = b;
  const std::int64_t product_64 = a_64 * b_64;
  if (product_64 > std::numeric_limits<std::int32_t>::max()) {
    error = true;
    return 0;
  } else if (product_64 < std::numeric_limits<std::int32_t>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<std::int32_t>(product_64);
  }
}
#endif

inline std::int32_t safe_division(std::int32_t a, std::int32_t b, il::io_t,
                                  bool& error) {
  if (b == 0 || (b == -1 && a == std::numeric_limits<std::int32_t>::min())) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

#ifdef UINT64_MAX
inline std::uint32_t safe_sum(std::uint32_t a, std::uint32_t b, il::io_t,
                              bool& error) {
  const std::uint64_t a_64 = a;
  const std::uint64_t b_64 = b;
  const std::uint64_t sum_64 = a_64 + b_64;
  if (sum_64 > std::numeric_limits<std::uint32_t>::max()) {
    error = true;
    return 0;
  } else if (sum_64 < std::numeric_limits<std::uint32_t>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<std::uint32_t>(sum_64);
  }
}
#endif

#ifdef UINT64_MAX
inline std::uint32_t safe_product(std::uint32_t a, std::uint32_t b, il::io_t,
                                  bool& error) {
  const std::uint64_t a_64 = a;
  const std::uint64_t b_64 = b;
  const std::uint64_t product_64 = a_64 * b_64;
  if (product_64 > std::numeric_limits<std::uint32_t>::max()) {
    error = true;
    return 0;
  } else if (product_64 < std::numeric_limits<std::uint32_t>::min()) {
    error = true;
    return 0;
  } else {
    return static_cast<std::uint32_t>(product_64);
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// 64-bit integer
////////////////////////////////////////////////////////////////////////////////

#ifdef INT64_MAX
inline std::int64_t safe_sum(std::int64_t a, std::int64_t b, il::io_t,
                             bool& error) {
  if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) ||
      (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b)) {
    error = true;
    return 0;
  } else {
    return a + b;
  }
}
#endif

#ifdef INT64_MAX
inline std::int64_t safe_difference(std::int64_t a, std::int64_t b, il::io_t,
                                    bool& error) {
  if ((b > 0 && a < std::numeric_limits<std::int64_t>::min() + b) ||
      (b < 0 && a > std::numeric_limits<std::int64_t>::max() + b)) {
    error = true;
    return 0;
  } else {
    return a - b;
  }
}
#endif

#ifdef INT64_MAX
inline std::int64_t safe_product(std::int64_t a, std::int64_t b, il::io_t,
                                 bool& error) {
  if (a > 0) {
    if (b > 0) {
      if (a > std::numeric_limits<std::int64_t>::max() / b) {
        error = true;
        return 0;
      }
    } else {
      if (b < std::numeric_limits<std::int64_t>::min() / a) {
        error = true;
        return 0;
      }
    }
  } else {
    if (b > 0) {
      if (a < std::numeric_limits<std::int64_t>::min() / b) {
        error = true;
        return 0;
      }
    } else {
      if (a != 0 && b < std::numeric_limits<std::int64_t>::max() / a) {
        error = true;
        return 0;
      }
    }
  }
  return a * b;
}
#endif

#ifdef INT64_MAX
inline std::int64_t safe_division(std::int64_t a, std::int64_t b, il::io_t,
                                  bool& error) {
  if (b == 0 || (b == -1 && a == std::numeric_limits<std::int64_t>::min())) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}
#endif

#ifdef UINT64_MAX
inline std::uint64_t safe_sum(std::uint64_t a, std::uint64_t b, il::io_t,
                              bool& error) {
  if (a > (std::numeric_limits<std::uint64_t>::max() - b)) {
    error = true;
    return 0;
  }
  return a + b;
}
#endif

#ifdef UINT64_MAX
inline std::uint64_t safe_product(std::uint64_t a, std::uint64_t b, il::io_t,
                                  bool& error) {
  if (b > 0) {
    if (a > std::numeric_limits<std::uint64_t>::max() / b) {
      error = true;
      return 0;
    }
  }
  error = error || false;
  return a * b;
}
#endif

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
T1 safe_convert(T2 n, il::io_t, bool& error) {
  error = true;
  return T1{};
};

#ifdef INT64_MAX
template <>
inline std::int64_t safe_convert(std::uint64_t n, il::io_t, bool& error) {
  if (n > std::numeric_limits<std::int64_t>::max()) {
    error = true;
    return 0;
  }
  return static_cast<std::int64_t>(n);
}
#endif

#ifdef INT64_MAX
template <>
inline std::uint64_t safe_convert(std::int64_t n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  }
  return static_cast<std::uint64_t>(n);
}
#endif
}
#endif  // IL_SAFE_ARITHMETIC_H
