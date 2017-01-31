//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Array.h>
#include <il/Array2D.h>

int main() {
  const il::int_t n = 127;
  il::Array2D<double> A{n, n, il::align, 32};

  return 0;
}
