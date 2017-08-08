//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/String.h>

int main() {
  il::String nom = "François Fayard";

  for (il::int_t i = 0; i < nom.size(); ++i) {
    std::cout << i << " " << nom.isRuneBoundary(i) << std::endl;
  }
  return 0;
}
