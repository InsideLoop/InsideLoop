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
  il::String s = u8"François";
  s.append(il::Rune::SmilingFaceWithHorns);

  std::cout << s.asCString() << std::endl;

  return 0;
}
