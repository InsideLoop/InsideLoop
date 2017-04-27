//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>
#include <il/Dynamic.h>

int main() {
  il::Dynamic x = 5000000000;
  if (x.is_int32()) {
    std::cout << "32 bits: " << x.to_int32() << std::endl;
  } else {
    std::cout << "64 bits: " << x.to_integer() << std::endl;

  }

  return 0;
}