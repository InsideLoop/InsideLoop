//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>
#include <string>

#include <il/String.h>

int main() {
  std::string s0 = "Francois";
  il::String s1 = il::toString(s0);

  return 0;
}
