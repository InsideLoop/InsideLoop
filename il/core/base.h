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

#define IL_INTEGER_PTRDIFF
//#define IL_INTEGER_INT

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

#ifdef IL_UNIT_TEST
#define IL_EXPECT_BOUND(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_BOUND(condition) ((void)0)
#else
#define IL_EXPECT_BOUND(condition) (condition) ? ((void)0) : std::abort()
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

#ifdef IL_INTEGER_INT
typedef int int_t;
typedef unsigned int uint_t;
const int_t int_t_max = std::numeric_limits<int>::max();
#else
typedef std::ptrdiff_t int_t;
typedef std::size_t uint_t;
const int_t int_t_max = std::numeric_limits<std::ptrdiff_t>::max();
#endif

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
struct default_value {
  static constexpr T value = T{};
};

template <>
struct default_value<bool> {
  static constexpr bool value = false;
};

template <>
struct default_value<char> {
  static constexpr char value = '\0';
};

template <>
struct default_value<std::int8_t> {
  static constexpr std::int8_t value = 123;
};

template <>
struct default_value<std::uint8_t> {
  static constexpr std::uint8_t value = 123;
};

template <>
struct default_value<std::int16_t> {
  static constexpr std::int16_t value = 12345;
};

template <>
struct default_value<std::uint16_t> {
  static constexpr std::uint16_t value = 12345;
};

template <>
struct default_value<std::int32_t> {
  static constexpr std::int32_t value = 1234567891;
};

template <>
struct default_value<std::uint32_t> {
  static constexpr std::uint32_t value = 1234567891;
};

#ifdef INT64_MAX
template <>
struct default_value<std::int64_t> {
  static constexpr std::int64_t value = 1234567891234567891;
};
#endif

#ifdef UINT64_MAX
template <>
struct default_value<std::uint64_t> {
  static constexpr std::uint64_t value = 1234567891234567891;
};
#endif

template <>
struct default_value<float> {
  static constexpr float value = std::numeric_limits<float>::quiet_NaN();
};

template <>
struct default_value<double> {
  static constexpr double value = std::numeric_limits<double>::quiet_NaN();
};

template <>
struct default_value<long double> {
  static constexpr long double value =
      std::numeric_limits<long double>::quiet_NaN();
};
}

#endif  // IL_BASE_H