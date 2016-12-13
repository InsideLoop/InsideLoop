//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstdio>

#include <il/linear_algebra/dot.h>

int main() {
  il::StaticArray3D<double, 6, 3, 9> A{};
  il::StaticArray<double, 9> B{};

  auto C = il::dot(A, B);
  IL_UNUSED(C);

  return 0;
}
