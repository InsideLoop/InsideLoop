//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/StaticArray2D.h>
#include <il/StaticArray2C.h>
#include <il/StaticArray3D.h>

int main() {
  il::StaticArray2D<double, 2, 3> A{il::value,
                                    {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}};
  il::StaticArray2C<double, 2, 3> B{il::value,
                                    {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}};
  il::StaticArray3D<double, 2, 3, 2> C{
      il::value,
      {{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
       {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}}};

  return 0;
}
