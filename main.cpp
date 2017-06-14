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
  il::String hello{};
  hello.append(il::CodePoint::grinning_face_with_smiling_eyes);

  std::cout << hello.as_c_string() << std::endl;

  return 0;
}
