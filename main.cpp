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
  il::String s = "François";

  il::String sub0 = s.prefix(4);
  il::String sub1 = s.suffix(3);
  il::String sub2 = s.substring(1, 4);

  return 0;
}
