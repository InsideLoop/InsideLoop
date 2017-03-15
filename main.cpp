//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Status.h>

int main() {
  il::Status status{};
  status.set_error(il::Error::matrix_singular);
  status.set_info("n", 10);
  status.set_info("rank", il::int_t{9});
  status.set_info("pi", 3.141459);
  status.set_info("file", "/home/fayard/Desktop");
  status.ignore_error();

  const double x = status.to_double("pi");
  IL_UNUSED(x);

  return 0;
}