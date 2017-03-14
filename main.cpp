//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================


#include <il/Info.h>

int main() {
  il::Info info{};
  info.set("line", 30);

  const int line = info.to_int("line");

  IL_UNUSED(line);

  return 0;
}