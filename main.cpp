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
  status.set(il::Error::matrix_singular);
  status.set("n", 10);
  status.set("rank", 9);
  status.ignore_error();

  return 0;
}