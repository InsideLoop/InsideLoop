//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Array2D.h>
#include <il/String.h>
#include <il/io/numpy.h>

int main() {
  il::Array<double> v{il::value, {1.0, 2.0, 3.0}};
  il::String filename = "/home/fayard/Desktop/b.npy";

  il::Status status{};
  il::save(v, filename, il::io, status);
  status.ignore_error();

  il::Array<double> w =
      il::load<il::Array<double>>(filename, il::io, status);
  status.ignore_error();

  return 0;
}
