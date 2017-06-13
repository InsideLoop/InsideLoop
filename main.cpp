//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/io/numpy.h>

int main() {
  il::String filename = "/Users/fayard/Desktop/a.npy";

  il::Array<double> v{10, 0.0};

  il::Status status{};
  il::save(v, filename, il::io, status);
  status.abort_on_error();

  return 0;
}
