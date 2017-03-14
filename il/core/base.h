//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_BASE_H
#define IL_BASE_H

// <cstddef> is needed for std::size_t and std::ptrdiff_t
#include <cstddef>
// <climits> is needed for LONG_MAX
#include <climits>
// <cstdlib> is needed for std::abort()
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////

//#define IL_BLAS_ATLAS

////////////////////////////////////////////////////////////////////////////////
// Assertions
////////////////////////////////////////////////////////////////////////////////

namespace il {
struct abort_exception {
  abort_exception() {
    (void)0;
  }
};

inline void abort() {
  std::abort();
}
}

// Use this when the expectation is fast to compute compared to the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_FAST(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_FAST(condition) ((void)0)
#else
#define IL_EXPECT_FAST(condition) (condition) ? ((void)0) : il::abort();
#endif

#ifdef NDEBUG
#define IL_EXPECT_FAST_NOTHROW(condition) ((void)0)
#else
#define IL_EXPECT_FAST_NOTHROW(condition) \
  (condition) ? ((void)0) : il::abort();
#endif

// Use this when the the expectation is as expensive to compute as the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_MEDIUM(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_MEDIUM(condition) ((void)0)
#else
#define IL_EXPECT_MEDIUM(condition) (condition) ? ((void)0) : il::abort();
#endif

// Use this when the the expectation is more expensive to compute than the
// function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_SLOW(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_SLOW(condition) ((void)0)
#else
#define IL_EXPECT_SLOW(condition) (condition) ? ((void)0) : il::abort();
#endif

// This one is not check and can contain code that is not run
#define IL_EXPECT_AXIOM(message) ((void)0)

#ifdef IL_UNIT_TEST
#define IL_ENSURE(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_ENSURE(condition) ((void)0)
#else
#define IL_ENSURE(condition) (condition) ? ((void)0) : il::abort();
#endif

#define IL_UNREACHABLE il::abort()

#define IL_UNUSED(var) ((void)var)

#ifndef NDEBUG
#define IL_DEFAULT_VALUE
#endif

////////////////////////////////////////////////////////////////////////////////
// Namespace il
////////////////////////////////////////////////////////////////////////////////

namespace il {

typedef std::ptrdiff_t int_t;

////////////////////////////////////////////////////////////////////////////////
// For arrays
////////////////////////////////////////////////////////////////////////////////

struct io_t {};
const io_t io{};
struct value_t {};
const value_t value{};
struct emplace_t {};
const emplace_t emplace{};
struct align_t {};
const align_t align{};

////////////////////////////////////////////////////////////////////////////////
// Default values for containers in debug mode
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct is_trivial {
  static constexpr bool value = false;
};

template <typename T>
T default_value() {
  return T{};
}

template <>
struct is_trivial<bool> {
  static constexpr bool value = true;
};

template <>
inline bool default_value<bool>() {
  return false;
}

template <>
struct is_trivial<char> {
  static constexpr bool value = true;
};

template <>
inline char default_value<char>() {
  return '\0';
}

template <>
struct is_trivial<signed char> {
  static constexpr bool value = true;
};

template <>
inline signed char default_value<signed char>() {
#if SCHAR_MAX == 127
  return 123;
#endif
}

template <>
struct is_trivial<unsigned char> {
  static constexpr bool value = true;
};

template <>
inline unsigned char default_value<unsigned char>() {
#if SCHAR_MAX == 127
  return 123;
#endif
}

template <>
struct is_trivial<short> {
  static constexpr bool value = true;
};

template <>
inline short default_value<short>() {
#if SHRT_MAX == 32767
  return 12345;
#endif
}

template <>
struct is_trivial<unsigned short> {
  static constexpr bool value = true;
};

template <>
inline unsigned short default_value<unsigned short>() {
#if SHRT_MAX == 32767
  return 12345;
#endif
}

template <>
struct is_trivial<int> {
  static constexpr bool value = true;
};

template <>
inline int default_value<int>() {
#if INT_MAX == 2147483647
  return 1234567891;
#endif
}

template <>
struct is_trivial<unsigned int> {
  static constexpr bool value = true;
};

template <>
inline unsigned int default_value<unsigned int>() {
#if INT_MAX == 2147483647
  return 1234567891;
#endif
}

template <>
struct is_trivial<long> {
  static constexpr bool value = true;
};

template <>
inline long default_value<long>() {
#if LONG_MAX == 2147483647
  return 1234567891;
#elif LONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct is_trivial<unsigned long> {
  static constexpr bool value = true;
};

template <>
inline unsigned long default_value<unsigned long>() {
#if LONG_MAX == 2147483647
  return 1234567891;
#elif LONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct is_trivial<long long> {
  static constexpr bool value = true;
};

template <>
inline long long default_value<long long>() {
#if LLONG_MAX == 2147483647
  return 1234567891;
#elif LLONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct is_trivial<unsigned long long> {
  static constexpr bool value = true;
};

template <>
inline unsigned long long default_value<unsigned long long>() {
#if LLONG_MAX == 2147483647
  return 1234567891;
#elif LLONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct is_trivial<float> {
  static constexpr bool value = true;
};

template <>
inline float default_value<float>() {
  return 0.0f / 0.0f;
}

template <>
struct is_trivial<double> {
  static constexpr bool value = true;
};

template <>
inline double default_value<double>() {
  return 0.0 / 0.0;
}

template <>
struct is_trivial<long double> {
  static constexpr bool value = true;
};

template <>
inline long double default_value<long double>() {
  return 0.0l / 0.0l;
}
}

#endif  // IL_BASE_H