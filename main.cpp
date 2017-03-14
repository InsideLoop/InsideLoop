//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================


#include <iostream>

#include <il/container/info/Info.h>

int main() {
  il::Info info{};
  info.set("line", il::int_t{30});
  info.set("inside", 3.14159);

  const il::int_t line = info.to_integer("line");

  IL_UNUSED(line);

  return 0;
}