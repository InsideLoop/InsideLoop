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
// <cstddef> is needed for integers of different sizes
#include <cstdint>
// <cstdlib> is needed for std::malloc, std::abort, etc
#include <cstdlib>
// <limits> is needed for std::numeric_limits
#include <limits>

////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////

#if INTPTR_MAX == INT32_MAX
#define IL_32_BIT_ENVIRONMENT
#elif INTPTR_MAX == INT64_MAX
#define IL_64_BIT_ENVIRONMENT
#endif

//#define IL_BLAS_ATLAS

////////////////////////////////////////////////////////////////////////////////
// Assertions
////////////////////////////////////////////////////////////////////////////////

namespace il {
struct abort_exception {};
}

// Use this when the expectation is fast to compute compared to the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_FAST(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_FAST(condition) ((void)0)
#else
#define IL_EXPECT_FAST(condition) (condition) ? ((void)0) : std::abort();
#endif

// Use this when the the expectation is as expensive to compute as the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_MEDIUM(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_MEDIUM(condition) ((void)0)
#else
#define IL_EXPECT_MEDIUM(condition) (condition) ? ((void)0) : std::abort();
#endif

// Use this when the the expectation is more expensive to compute than the
// function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_SLOW(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_SLOW(condition) ((void)0)
#else
#define IL_EXPECT_SLOW(condition) (condition) ? ((void)0) : std::abort();
#endif

#define IL_EXPECT_AXIOM(message) ((void)0)

// This one is not check and can contain code that is not run
#ifdef IL_UNIT_TEST
#define IL_ENSURE(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_ENSURE(condition) ((void)0)
#else
#define IL_ENSURE(condition) (condition) ? ((void)0) : std::abort();
#endif

#define IL_UNREACHABLE std::abort()

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

#define IL_SIMD 32
const short simd = IL_SIMD;
#define IL_CACHELINE 64
const short cacheline = IL_CACHELINE;
const short page = 4096;

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
struct is_trivial<std::int8_t> {
  static constexpr bool value = true;
};

template <>
inline std::int8_t default_value<std::int8_t>() {
  return 123;
}

template <>
struct is_trivial<std::uint8_t> {
  static constexpr bool value = true;
};

template <>
inline std::uint8_t default_value<std::uint8_t>() {
  return 123;
}

template <>
struct is_trivial<std::int16_t> {
  static constexpr bool value = true;
};

template <>
inline std::int16_t default_value<std::int16_t>() {
  return 12345;
}

template <>
struct is_trivial<std::uint16_t> {
  static constexpr bool value = true;
};

template <>
inline std::uint16_t default_value<std::uint16_t>() {
  return 12345;
}

template <>
struct is_trivial<std::int32_t> {
  static constexpr bool value = true;
};

template <>
inline std::int32_t default_value<std::int32_t>() {
  return 1234567891;
}

template <>
struct is_trivial<std::uint32_t> {
  static constexpr bool value = true;
};

template <>
inline std::uint32_t default_value<std::uint32_t>() {
  return 1234567891;
}

#ifdef INT64_MAX
template <>
struct is_trivial<std::int64_t> {
  static constexpr bool value = true;
};

template <>
inline std::int64_t default_value<std::int64_t>() {
  return 1234567891234567891;
}
#endif

#ifdef UINT64_MAX
template <>
struct is_trivial<std::uint64_t> {
  static constexpr bool value = true;
};

template <>
inline std::uint64_t default_value<std::uint64_t>() {
  return 1234567891234567891;
}
#endif

template <>
struct is_trivial<float> {
  static constexpr bool value = true;
};

template <>
inline float default_value<float>() {
  return std::numeric_limits<float>::quiet_NaN();
}

template <>
struct is_trivial<double> {
  static constexpr bool value = true;
};

template <>
inline double default_value<double>() {
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
struct is_trivial<long double> {
  static constexpr bool value = true;
};

template <>
inline long double default_value<long double>() {
  return std::numeric_limits<long double>::quiet_NaN();
}

}

#endif  // IL_BASE_H