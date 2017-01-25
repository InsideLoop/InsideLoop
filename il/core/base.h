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
#define IL_EXPECT_FAST(condition) \
  (condition) ? ((void) 0) : ::abort();
#endif


// Use this when the the expectation is as expensive to compute as the function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_MEDIUM(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_MEDIUM(condition) ((void)0)
#else
#define IL_EXPECT_MEDIUM(condition) \
  (condition) ? ((void) 0) : ::abort();
#endif

// Use this when the the expectation is more expensive to compute than the
// function
#ifdef IL_UNIT_TEST
#define IL_EXPECT_SLOW(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_SLOW(condition) ((void)0)
#else
#define IL_EXPECT_SLOW(condition) \
  (condition) ? ((void) 0) : ::abort();
#endif

#ifdef IL_UNIT_TEST
#define IL_EXPECT_BOUND(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_EXPECT_BOUND(condition) ((void)0)
#else
#define IL_EXPECT_BOUND(condition) (condition) ? ((void)0) : abort()
#endif



#define IL_EXPECT_AXIOM(message) (void)0

// This one is not check and can contain code that is not run
#ifdef IL_UNIT_TEST
#define IL_ENSURE(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_ENSURE(condition) ((void)0)
#else
#define IL_ENSURE(condition) \
  (condition) ? ((void) 0) : ::abort();
#endif


//#ifndef NDEBUG
//#define IL_DEBUG_VISUALIZER
//#endif

#ifndef NDEBUG
#define IL_DEFAULT_VALUE
#endif

#ifndef NDEBUG
#define IL_INVARIANCE
#endif

#ifdef IL_UNIT_TEST
#define IL_ASSERT(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#else
#define IL_ASSERT(condition) (condition) ? ((void)0) : ::abort()
#endif

#ifdef IL_UNIT_TEST
#define IL_ASSERT_PRECOND(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#else
#define IL_ASSERT_PRECOND(condition) (condition) ? ((void)0) : ::abort()
#endif

// IL_EXPECT_BOUND is used for bounds checking in Array containers
// - in debug mode, the program is aborted
// - in release mode, no bounds checking is done
// - in unit test mode, an exception is thrown so our unit test can check that
//   bounds checking is done correctly
//
#define IL_UNUSED(var) (void)var

#define IL_UNREACHABLE abort()

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
T default_value() {
  return T{};
}

template <>
inline float default_value<float>() {
  return std::numeric_limits<float>::quiet_NaN();
}

template <>
inline double default_value<double>() {
  return std::numeric_limits<double>::quiet_NaN();
}

template <>
inline long double default_value<long double>() {
  return std::numeric_limits<long double>::quiet_NaN();
}

template <>
inline char default_value<char>() {
  return 123;
}

template <>
inline unsigned char default_value<unsigned char>() {
  return 123;
}

template <>
inline int default_value<int>() {
  return 1234567891;
}

template <>
inline unsigned int default_value<unsigned int>() {
  return 1234567891;
}

template <>
inline long int default_value<long int>() {
  return (sizeof(long int) == 8) ? 1234567891234567891 : 1234567891;
}

template <>
inline unsigned long int default_value<unsigned long int>() {
  return (sizeof(long int) == 8) ? 1234567891234567891 : 1234567891;
}

}

#endif  // IL_BASE_H