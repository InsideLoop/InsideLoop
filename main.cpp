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
#include <il/Timer.h>
#include <il/unicode.h>

int main() {
  const il::int_t n = 30;
  il::Timer timer{};

  timer.start();
  std::string s1 = "essai";
  std::string s1init = "essai";
  for (il::int_t k = 0; k < n; ++k) {
    s1.append(s1init);
  }
  timer.stop();

  std::cout << "std::string: " << timer.elapsed() << " s" << std::endl;

  timer.reset();
  timer.start();
  il::String s2 = "essai";
  il::String s2init = "essai";
  s2init.append(il::CodePoint::smiling_face_with_horns);
  for (il::int_t k = 0; k < n; ++k) {
    s2.append(s2init);
  }
  timer.stop();

  std::cout << "il::String: " << timer.elapsed() << " s" << std::endl;

  return 0;
}
