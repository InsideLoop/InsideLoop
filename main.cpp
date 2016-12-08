//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Array4D.h>

int main() {
  const int n = 2;
  il::Array4D<double> A{n, n, n, n};
  for (int i3 = 0; i3 < A.size(3); ++i3) {
    for (int i2 = 0; i2 < A.size(2); ++i2) {
      for (int i1 = 0; i1 < A.size(1); ++i1) {
        for (int i0 = 0; i0 < A.size(0); ++i0) {
          A(i0, i1, i2, i3) = 1000 * i0 + 100 * i1 + 10 * i2 + i3;
        }
      }
    }
  }

  return 0;
}