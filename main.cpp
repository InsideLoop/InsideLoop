//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Array3D.h>

int main() {
  const il::int_t n = 10;
  il::Array3D<double> A{n, n, n, il::align, 32};

  return 0;
}
