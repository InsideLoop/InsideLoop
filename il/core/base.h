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
// Multiple platforms
////////////////////////////////////////////////////////////////////////////////

#if defined(WIN32) || defined(_WIN32) || \
    defined(__WIN32) && !defined(__CYGWIN__)
#define IL_WINDOWS
#else
#define IL_UNIX
#endif

//#define IL_32_BIT
#define IL_64_BIT

////////////////////////////////////////////////////////////////////////////////
// Assertions
////////////////////////////////////////////////////////////////////////////////

namespace il {
struct AbortException {
  AbortException() { (void)0; }
};

inline void abort() { std::abort(); }

}  // namespace il

// Use this when the expectation is fast to compute compared to the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_FAST(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_EXPECT_FAST(condition) ((void)0)
#else
#define IL_EXPECT_FAST(condition) (condition) ? ((void)0) : il::abort();
#endif

#ifdef NDEBUG
#define IL_EXPECT_FAST_NOTHROW(condition) ((void)0)
#else
#define IL_EXPECT_FAST_NOTHROW(condition) (condition) ? ((void)0) : il::abort();
#endif

// Use this when the the expectation is as expensive to compute as the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_MEDIUM(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_EXPECT_MEDIUM(condition) ((void)0)
#else
#define IL_EXPECT_MEDIUM(condition) (condition) ? ((void)0) : il::abort();
#endif

// Use this when the the expectation is more expensive to compute than the
// function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_SLOW(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
#elif NDEBUG
#define IL_EXPECT_SLOW(condition) ((void)0)
#else
#define IL_EXPECT_SLOW(condition) (condition) ? ((void)0) : il::abort();
#endif

// This one is not check and can contain code that is not run
#define IL_EXPECT_AXIOM(message) ((void)0)

#ifdef IL_UNIT_TEST
#define IL_ENSURE(condition) \
  (condition) ? ((void)0) : throw il::AbortException {}
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

#ifdef IL_64_BIT
#define IL_INTEGER_MAX 9223372036854775807
#else
#define IL_INTEGER_MAX 2147483647
#endif

template <typename T>
T max(T a, T b) {
  return a >= b ? a : b;
}

template <typename T>
T min(T a, T b) {
  return a <= b ? a : b;
}

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
struct isTrivial {
  static constexpr bool value = false;
};

template <typename T>
T defaultValue() {
  return T{};
}

template <>
struct isTrivial<bool> {
  static constexpr bool value = true;
};

template <>
inline bool defaultValue<bool>() {
  return false;
}

template <>
struct isTrivial<char> {
  static constexpr bool value = true;
};

template <>
inline char defaultValue<char>() {
  return '\0';
}

template <>
struct isTrivial<signed char> {
  static constexpr bool value = true;
};

template <>
inline signed char defaultValue<signed char>() {
#if SCHAR_MAX == 127
  return 123;
#endif
}

template <>
struct isTrivial<unsigned char> {
  static constexpr bool value = true;
};

template <>
inline unsigned char defaultValue<unsigned char>() {
#if SCHAR_MAX == 127
  return 123;
#endif
}

template <>
struct isTrivial<short> {
  static constexpr bool value = true;
};

template <>
inline short defaultValue<short>() {
#if SHRT_MAX == 32767
  return 12345;
#endif
}

template <>
struct isTrivial<unsigned short> {
  static constexpr bool value = true;
};

template <>
inline unsigned short defaultValue<unsigned short>() {
#if SHRT_MAX == 32767
  return 12345;
#endif
}

template <>
struct isTrivial<int> {
  static constexpr bool value = true;
};

template <>
inline int defaultValue<int>() {
#if INT_MAX == 2147483647
  return 1234567891;
#endif
}

template <>
struct isTrivial<unsigned int> {
  static constexpr bool value = true;
};

template <>
inline unsigned int defaultValue<unsigned int>() {
#if INT_MAX == 2147483647
  return 1234567891;
#endif
}

template <>
struct isTrivial<long> {
  static constexpr bool value = true;
};

template <>
inline long defaultValue<long>() {
#if LONG_MAX == 2147483647
  return 1234567891;
#elif LONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<unsigned long> {
  static constexpr bool value = true;
};

template <>
inline unsigned long defaultValue<unsigned long>() {
#if LONG_MAX == 2147483647
  return 1234567891;
#elif LONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<long long> {
  static constexpr bool value = true;
};

template <>
inline long long defaultValue<long long>() {
#if LLONG_MAX == 2147483647
  return 1234567891;
#elif LLONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<unsigned long long> {
  static constexpr bool value = true;
};

template <>
inline unsigned long long defaultValue<unsigned long long>() {
#if LLONG_MAX == 2147483647
  return 1234567891;
#elif LLONG_MAX == 9223372036854775807
  return 1234567891234567891;
#endif
}

template <>
struct isTrivial<float> {
  static constexpr bool value = true;
};

template <>
inline float defaultValue<float>() {
#ifdef IL_UNIX
  return 0.0f / 0.0f;
#else
  return 0.0f;
#endif
}

template <>
struct isTrivial<double> {
  static constexpr bool value = true;
};

template <>
inline double defaultValue<double>() {
#ifdef IL_UNIX
  return 0.0 / 0.0;
#else
  return 0.0;
#endif
}

template <>
struct isTrivial<long double> {
  static constexpr bool value = true;
};

template <>
inline long double defaultValue<long double>() {
#ifdef IL_UNIX
  return 0.0l / 0.0l;
#else
  return 0.0l;
#endif
}
}  // namespace il

#endif  // IL_BASE_H
