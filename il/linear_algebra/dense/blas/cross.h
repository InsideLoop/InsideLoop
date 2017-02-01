//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_CROSS_H
#define IL_CROSS_H

#include <il/StaticArray.h>

namespace il {

template <typename T>
il::StaticArray<T, 2> cross(const il::StaticArray<T, 2>& x) {
  il::StaticArray<T, 2> ans{};
  ans[0] = -x[1];
  ans[1] = x[0];
  return ans;
}

template <typename T>
il::StaticArray<T, 3> cross(const il::StaticArray<T, 3>& x,
                            const il::StaticArray<T, 3>& y) {
  il::StaticArray<T, 3> ans{};
  ans[0] = x[1] * y[2] - x[2] * y[1];
  ans[1] = x[2] * y[0] - x[0] * y[2];
  ans[2] = x[0] * y[1] - x[1] * y[0];
  return ans;
}

}

#endif  // IL_CROSS_H
