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

namespace il {

////////////////////////////////////////////////////////////////////////////////
// 32-bit integer
////////////////////////////////////////////////////////////////////////////////

inline std::int32_t safe_sum(std::int32_t a, std::int32_t b, il::io_t,
                             bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::int32_t ans;
  error = __builtin_sadd_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > std::numeric_limits<std::int32_t>::max() - b
            : a < std::numeric_limits<std::int32_t>::min() - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline std::int32_t safe_difference(std::int32_t a, std::int32_t b, il::io_t,
                                    bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::int32_t ans;
  error = __builtin_ssub_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < std::numeric_limits<std::int32_t>::min() + b
            : a > std::numeric_limits<std::int32_t>::max() + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline std::int32_t safe_product(std::int32_t a, std::int32_t b, il::io_t,
                                 bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::int32_t ans;
  error = __builtin_smul_overflow(a, b, &ans);
  return ans;
#elif defined(IL_64_BIT_ENVIRONMENT)
  const std::int64_t a_64 = a;
  const std::int64_t b_64 = b;
  const std::int64_t product_64 = a_64 * b_64;
  if (product_64 > std::numeric_limits<std::int32_t>::max() ||
      product_64 < std::numeric_limits<std::int32_t>::min()) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<std::int32_t>(product_64);
  }
#else
  if (b > 0
          ? a > std::numeric_limits<std::int32_t>::max() / b ||
                a < std::numeric_limits<std::int32_t>::min() / b
          : (b < -1
                 ? a > std::numeric_limits<std::int32_t>::min() / b ||
                       a < std::numeric_limits<std::int32_t>::max() / b
                 : b == -1 && a == std::numeric_limits<std::int32_t>::min())) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline std::int32_t safe_product(std::int32_t a, std::int32_t b, std::int32_t c,
                                 il::io_t, bool& error) {
  bool error_first;
  const std::int32_t ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const std::int32_t abc = il::safe_product(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline std::int32_t safe_product(std::int32_t a, std::int32_t b, std::int32_t c,
                                 std::int32_t d, il::io_t, bool& error) {
  bool error_first;
  const std::int32_t ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const std::int32_t cd = il::safe_product(c, d, il::io, error_second);
  bool error_third;
  const std::int32_t abcd = il::safe_product(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline std::int32_t safe_division(std::int32_t a, std::int32_t b, il::io_t,
                                  bool& error) {
  if (b == 0 || (b == -1 && a == std::numeric_limits<std::int32_t>::min())) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline std::uint32_t safe_sum(std::uint32_t a, std::uint32_t b, il::io_t,
                              bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::uint32_t ans;
  error = __builtin_uadd_overflow(a, b, &ans);
  return ans;
#else
  if (a > std::numeric_limits<std::uint32_t>::max() - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline std::uint32_t safe_product(std::uint32_t a, std::uint32_t b, il::io_t,
                                  bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::uint32_t ans;
  error = __builtin_umul_overflow(a, b, &ans);
  return ans;
#elif defined(IL_64_BIT_ENVIRONMENT)
  const std::uint64_t a_64 = a;
  const std::uint64_t b_64 = b;
  const std::uint64_t product_64 = a_64 * b_64;
  if (product_64 > std::numeric_limits<std::uint32_t>::max()) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<std::uint32_t>(product_64);
  }
#else
  if (b > 0 && a > std::numeric_limits<std::int32_t>::max() / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// 64-bit integer
////////////////////////////////////////////////////////////////////////////////

#ifdef IL_64_BIT_ENVIRONMENT
inline std::int64_t safe_sum(std::int64_t a, std::int64_t b, il::io_t,
                             bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::int64_t ans;
  error = __builtin_saddll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > std::numeric_limits<std::int64_t>::max() - b
            : a < std::numeric_limits<std::int64_t>::min() - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
inline std::int64_t safe_difference(std::int64_t a, std::int64_t b, il::io_t,
                                    bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::int64_t ans;
  error = __builtin_ssubll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < std::numeric_limits<std::int64_t>::min() + b
            : a > std::numeric_limits<std::int64_t>::max() + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
inline std::int64_t safe_product(std::int64_t a, std::int64_t b, il::io_t,
                                 bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::int64_t ans;
  error = __builtin_smulll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0
          ? a > std::numeric_limits<std::int64_t>::max() / b ||
                a < std::numeric_limits<std::int64_t>::min() / b
          : (b < -1
                 ? a > std::numeric_limits<std::int64_t>::min() / b ||
                       a < std::numeric_limits<std::int64_t>::max() / b
                 : b == -1 && a == std::numeric_limits<std::int64_t>::min())) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
inline std::int64_t safe_product(std::int64_t a, std::int64_t b, std::int64_t c,
                                 il::io_t, bool& error) {
  bool error_first;
  const std::int64_t ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const std::int64_t abc = il::safe_product(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
inline std::int64_t safe_product(std::int64_t a, std::int64_t b, std::int64_t c,
                                 std::int64_t d, il::io_t, bool& error) {
  bool error_first;
  const std::int64_t ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const std::int64_t cd = il::safe_product(c, d, il::io, error_second);
  bool error_third;
  const std::int64_t abcd = il::safe_product(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
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

#ifdef IL_64_BIT_ENVIRONMENT
inline std::uint64_t safe_sum(std::uint64_t a, std::uint64_t b, il::io_t,
                              bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::uint64_t ans;
  error = __builtin_uaddll_overflow(a, b, &ans);
  return ans;
#else
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
inline std::uint64_t safe_product(std::uint64_t a, std::uint64_t b, il::io_t,
                                  bool& error) {
#if __GNUC__ >= 6 || \
    (__clang_major__ >= 4 || (__clang__major__ >= 3 && __clang_minor_ >= 9))
  std::uint64_t ans;
  error = __builtin_umulll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 && a > std::numeric_limits<std::int64_t>::max() / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}
#endif

////////////////////////////////////////////////////////////////////////////////
// rounding
////////////////////////////////////////////////////////////////////////////////

inline std::int32_t safe_upper_round(std::int32_t a, std::int32_t b, il::io_t,
                                     bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const std::uint32_t q =
      static_cast<std::uint32_t>(a) / static_cast<std::uint32_t>(b);
  const std::uint32_t r =
      static_cast<std::uint32_t>(a) % static_cast<std::uint32_t>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const std::int32_t q_plus_one =
        il::safe_sum(static_cast<std::int32_t>(q), static_cast<std::int32_t>(1),
                     il::io, error_sum);
    const std::int32_t ans =
        il::safe_product(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}

#ifdef IL_64_BIT_ENVIRONMENT
inline std::int64_t safe_upper_round(std::int64_t a, std::int64_t b, il::io_t,
                                     bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const std::uint64_t q =
      static_cast<std::uint64_t>(a) / static_cast<std::uint64_t>(b);
  const std::uint64_t r =
      static_cast<std::uint64_t>(a) % static_cast<std::uint64_t>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const std::int64_t q_plus_one =
        il::safe_sum(static_cast<std::int64_t>(q), static_cast<std::int64_t>(1),
                     il::io, error_sum);
    const std::int64_t ans =
        il::safe_product(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
// convert
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
T1 safe_convert(T2 n, il::io_t, bool& error) {
  IL_UNREACHABLE;
}

template <>
inline std::int32_t safe_convert(std::uint32_t n, il::io_t, bool& error) {
  if (n > std::numeric_limits<std::int32_t>::max()) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<std::int32_t>(n);
  }
}

template <>
inline std::uint32_t safe_convert(std::int32_t n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<std::uint32_t>(n);
  }
}

#ifdef IL_64_BIT_ENVIRONMENT
template <>
inline std::int64_t safe_convert(std::uint64_t n, il::io_t, bool& error) {
  if (n > std::numeric_limits<std::int64_t>::max()) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<std::int64_t>(n);
  }
}
#endif

#ifdef IL_64_BIT_ENVIRONMENT
template <>
inline std::uint64_t safe_convert(std::int64_t n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<std::uint64_t>(n);
  }
}
#endif
}

#endif  // IL_SAFE_ARITHMETIC_H
