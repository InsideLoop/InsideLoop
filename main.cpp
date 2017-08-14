//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/String.h>
#include <string>
#include <iostream>

int main() {
  il::String a = "François";
  il::String b = "Fayard";

  il::String c = il::join(a, " ", b);

  return 0;
}
