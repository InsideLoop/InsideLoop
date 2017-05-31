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
  il::String string{};
  string.append(il::CodePoint::greek_small_letter_alpha);
  string.append(il::CodePoint::greek_small_letter_beta);
  string.append(" = 0");

  std::cout << string.c_string() << std::endl;

  return 0;
}