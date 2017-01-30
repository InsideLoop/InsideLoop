//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/core/math/safe_arithmetic.h>

int main() {
  const int a = std::numeric_limits<int>::max();
  const int b = 0;
  bool error = false;
  const int c = il::safe_sum(a, b, il::io, error);

  std::cout << c << " " << error << std::endl;

  return 0;
}
