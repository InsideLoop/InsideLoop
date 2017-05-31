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

#include <il/base.h>

namespace il {

////////////////////////////////////////////////////////////////////////////////
// int
////////////////////////////////////////////////////////////////////////////////

inline int safe_sum(int a, int b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  int ans;
  error = __builtin_sadd_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > INT_MAX - b : a < INT_MIN - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline int safe_difference(int a, int b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  int ans;
  error = __builtin_ssub_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < INT_MIN + b : a > INT_MAX + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline int safe_product(int a, int b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  int ans;
  error = __builtin_smul_overflow(a, b, &ans);
  return ans;
#elif INT_MAX < LLONG_MAX
  const long long a_llong = a;
  const long long b_llong = b;
  const long long product_llong = a_llong * b_llong;
  if (product_llong > INT_MAX || product_llong < INT_MIN) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<int>(product_llong);
  }
#else
  if (b > 0 ? a > INT_MAX / b || a < INT_MIN / b
            : (b < -1 ? a > INT_MIN / b || a < INT_MAX / b
                      : b == -1 && a == INT_MIN)) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline int safe_product(int a, int b, int c, il::io_t, bool& error) {
  bool error_first;
  const int ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const int abc = il::safe_product(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline int safe_product(int a, int b, int c, int d, il::io_t, bool& error) {
  bool error_first;
  const int ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const int cd = il::safe_product(c, d, il::io, error_second);
  bool error_third;
  const int abcd = il::safe_product(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline int safe_division(int a, int b, il::io_t, bool& error) {
  if (b == 0 || (b == -1 && a == INT_MIN)) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline unsigned safe_sum(unsigned a, unsigned b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned ans;
  error = __builtin_uadd_overflow(a, b, &ans);
  return ans;
#else
  if (a > UINT_MAX - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline unsigned safe_product(unsigned a, unsigned b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned ans;
  error = __builtin_umul_overflow(a, b, &ans);
  return ans;
#elif INT_MAX < LONG_MAX
  const unsigned long a_long = a;
  const unsigned long b_long = b;
  const unsigned long product_long = a_long * b_long;
  if (product_long > UINT_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned>(product_long);
  }
#else
  if (b > 0 && a > INT_MAX / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// long
////////////////////////////////////////////////////////////////////////////////

inline long safe_sum(long a, long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long ans;
  error = __builtin_saddl_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LONG_MAX - b : a < LONG_MIN - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline long safe_difference(long a, long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long ans;
  error = __builtin_ssubl_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < LONG_MIN + b : a > LONG_MAX + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline long safe_product(long a, long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long ans;
  error = __builtin_smull_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LONG_MAX / b || a < LONG_MIN / b
            : (b < -1 ? a > LONG_MIN / b || a < LONG_MAX / b
                      : b == -1 && a == LONG_MIN)) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline long safe_product(long a, long b, long c, il::io_t, bool& error) {
  bool error_first;
  const long ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const long abc = il::safe_product(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline long safe_product(long a, long b, long c, long d, il::io_t,
                         bool& error) {
  bool error_first;
  const long ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const long cd = il::safe_product(c, d, il::io, error_second);
  bool error_third;
  const long abcd = il::safe_product(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline long safe_division(long a, long b, il::io_t, bool& error) {
  if (b == 0 || (b == -1 && a == LONG_MIN)) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline unsigned long safe_sum(unsigned long a, unsigned long b, il::io_t,
                              bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long ans;
  error = __builtin_uaddl_overflow(a, b, &ans);
  return ans;
#else
  if (a > ULONG_MAX - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline unsigned long safe_product(unsigned long a, unsigned long b, il::io_t,
                                  bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long ans;
  error = __builtin_umull_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 && a > ULONG_MAX / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// long long
////////////////////////////////////////////////////////////////////////////////

inline long long safe_sum(long long a, long long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long long ans;
  error = __builtin_saddll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LLONG_MAX - b : a < LLONG_MIN - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline long long safe_difference(long long a, long long b, il::io_t,
                                 bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long long ans;
  error = __builtin_ssubll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a < LLONG_MIN + b : a > LLONG_MAX + b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a - b;
  }
#endif
}

inline long long safe_product(long long a, long long b, il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  long long ans;
  error = __builtin_smulll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 ? a > LLONG_MAX / b || a < LLONG_MIN / b
            : (b < -1 ? a > LLONG_MIN / b || a < LLONG_MAX / b
                      : b == -1 && a == LLONG_MIN)) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

inline long long safe_product(long long a, long long b, long long c, il::io_t,
                              bool& error) {
  bool error_first;
  const long long ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const long long abc = il::safe_product(ab, c, il::io, error_second);
  if (error_first || error_second) {
    error = true;
    return 0;
  } else {
    error = false;
    return abc;
  }
}

inline long long safe_product(long long a, long long b, long long c,
                              long long d, il::io_t, bool& error) {
  bool error_first;
  const long long ab = il::safe_product(a, b, il::io, error_first);
  bool error_second;
  const long long cd = il::safe_product(c, d, il::io, error_second);
  bool error_third;
  const long long abcd = il::safe_product(ab, cd, il::io, error_third);
  if (error_first || error_second || error_third) {
    error = true;
    return 0;
  } else {
    error = false;
    return abcd;
  }
}

inline long long safe_division(long long a, long long b, il::io_t,
                               bool& error) {
  if (b == 0 || (b == -1 && a == LLONG_MIN)) {
    error = true;
    return 0;
  } else {
    return a / b;
  }
}

inline unsigned long long safe_sum(unsigned long long a, unsigned long long b,
                                   il::io_t, bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long long ans;
  error = __builtin_uaddll_overflow(a, b, &ans);
  return ans;
#else
  if (a > ULLONG_MAX - b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a + b;
  }
#endif
}

inline unsigned long long safe_product(unsigned long long a,
                                       unsigned long long b, il::io_t,
                                       bool& error) {
#if IL_BUILTIN_SAFE_ARITHMETIC
  unsigned long long ans;
  error = __builtin_umulll_overflow(a, b, &ans);
  return ans;
#else
  if (b > 0 && a > ULLONG_MAX / b) {
    error = true;
    return 0;
  } else {
    error = false;
    return a * b;
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////
// rounding
////////////////////////////////////////////////////////////////////////////////

inline int safe_upper_round(int a, int b, il::io_t, bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const unsigned q = static_cast<unsigned>(a) / static_cast<unsigned>(b);
  const unsigned r = static_cast<unsigned>(a) % static_cast<unsigned>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const int q_plus_one = il::safe_sum(static_cast<int>(q),
                                        static_cast<int>(1), il::io, error_sum);
    const int ans = il::safe_product(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}

inline long safe_upper_round(long a, long b, il::io_t, bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const unsigned long q =
      static_cast<unsigned long>(a) / static_cast<unsigned long>(b);
  const unsigned long r =
      static_cast<unsigned long>(a) % static_cast<unsigned long>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const long q_plus_one = il::safe_sum(
        static_cast<long>(q), static_cast<long>(1), il::io, error_sum);
    const long ans = il::safe_product(q_plus_one, b, il::io, error_product);
    if (error_sum || error_product) {
      error = true;
      return 0;
    } else {
      error = false;
      return ans;
    }
  }
}

inline long long safe_upper_round(long long a, long long b, il::io_t,
                                  bool& error) {
  IL_EXPECT_FAST(a >= 0);
  IL_EXPECT_FAST(b > 0);

  const unsigned long long q =
      static_cast<unsigned long long>(a) / static_cast<unsigned long long>(b);
  const unsigned long long r =
      static_cast<unsigned long long>(a) % static_cast<unsigned long long>(b);
  if (r == 0) {
    error = false;
    return a;
  } else {
    bool error_sum = false;
    bool error_product = false;
    const long long q_plus_one =
        il::safe_sum(static_cast<long long>(q), static_cast<long long>(1),
                     il::io, error_sum);
    const long long ans =
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

////////////////////////////////////////////////////////////////////////////////
// convert
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2>
T1 safe_convert(T2 n, il::io_t, bool& error) {
  IL_UNUSED(n);
  IL_UNUSED(error);
  IL_UNREACHABLE;
}

template <>
inline int safe_convert(unsigned n, il::io_t, bool& error) {
  if (n > INT_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<int>(n);
  }
}

template <>
inline unsigned safe_convert(int n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned>(n);
  }
}

template <>
inline long safe_convert(unsigned long n, il::io_t, bool& error) {
  if (n > LONG_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<long>(n);
  }
}

template <>
inline unsigned long safe_convert(long n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned long>(n);
  }
}

template <>
inline long long safe_convert(unsigned long long n, il::io_t, bool& error) {
  if (n > LLONG_MAX) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<long long>(n);
  }
}

template <>
inline unsigned long long safe_convert(long long n, il::io_t, bool& error) {
  if (n < 0) {
    error = true;
    return 0;
  } else {
    error = false;
    return static_cast<unsigned long long>(n);
  }
}
}  // namespace il

#endif  // IL_SAFE_ARITHMETIC_H
