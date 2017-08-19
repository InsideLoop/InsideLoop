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
  il::String s0 = "Francois";
  il::String s1 = "Fayard";

  il::String s = il::join(s0, " ", s1, s0, s1, "f", s0, s1);

  return 0;
}
