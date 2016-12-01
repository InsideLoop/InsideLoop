//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/StaticArray3D.h>

int main() {
  il::StaticArray3D<double, 2, 3, 4> A{};
  A(0, 0, 0) = 0.0;
  A(1, 1, 1) = 0.0;
  A(1, 2, 3) = 0.0;

  return 0;
}
