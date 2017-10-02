//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Array2C.h>
#include <il/Array3D.h>
#include <il/Array3C.h>
#include <il/String.h>

int main() {
  il::Array<il::int_t> z{il::value, {1, 2, 3}};
  il::Array<double> a{il::value, {1.0, 2.0, 3.0}};
  il::Array2D<double> b{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  il::Array2C<double> c{il::value, {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  il::Array3D<double> d{il::value, {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                    {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}}}};
  il::Array3C<double> e{il::value, {{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}},
                                    {{11.0, 12.0, 13.0}, {14.0, 15.0, 16.0}}}};

  return 0;
}
