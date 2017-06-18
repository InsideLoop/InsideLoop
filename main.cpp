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
#include <il/unicode.h>

int main() {
  il::String name = u8"François ";
  name.append(il::UnicodeScalar::kGrinningFaceWithSmilingEyes);

  std::cout << name.asCString() << std::endl;

  return 0;
}
