//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_ILASSERT_H
#define IL_ILASSERT_H

#include <cstdlib>

namespace il {
struct abort_exception {};
}

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

// IL_ASSERT_BOUNDS is used for bounds checking in Array containers
// - in debug mode, the program is aborted
// - in release mode, no bounds checking is done
// - in unit test mode, an exception is thrown so our unit test can check that
//   bounds checking is done correctly
//
#ifdef IL_UNIT_TEST
#define IL_ASSERT_BOUNDS(condition) \
  (condition) ? ((void)0) : throw il::abort_exception {}
#elif NDEBUG
#define IL_ASSERT_BOUNDS(condition) ((void)0)
#else
#define IL_ASSERT_BOUNDS(condition) (condition) ? ((void)0) : abort()
#endif

#define IL_UNUSED(var) (void)var

#endif  // IL_ILASSERT_H
