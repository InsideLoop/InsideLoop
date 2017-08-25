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
  il::String s = "Hello!Hello!";
  il::StringView v = s;

  IL_UNUSED(v);
  IL_UNUSED(s);

  return 0;
}
