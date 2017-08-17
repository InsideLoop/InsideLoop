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
#include <il/Array.h>

int main() {
  il::Array<il::String> s{10, "François"};
  IL_UNUSED(s);

  return 0;
}
