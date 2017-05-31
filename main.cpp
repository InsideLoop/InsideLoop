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
  string.append(il::CodePoint::for_all);
  string.append("x");
  string.append(il::CodePoint::element_of);
  string.append(il::CodePoint::double_struck_capital_r);

  std::cout << "String: " << string.c_string() << std::endl;
  std::cout << "Length: " << string.length() << std::endl;
  std::cout << "Size: " << string.size() << std::endl;
  std::cout << "Capacity: " << string.capacity() << std::endl;

  std::cout << std::endl;

  for (il::int_t i = 0; i < string.size(); i = string.next_cp(i)) {
    std::cout << "i: " << i << ", codepoint: " << string.to_cp(i) << std::endl;
  }

  return 0;
}