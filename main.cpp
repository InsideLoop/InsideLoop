//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/SparseArray2C.h>
#include <il/linear_algebra/sparse/blas/sparse_dot.h>

#include <il/linear_algebra/sparse/factorization/_test/matrix/heat.h>

int main() {
  const int n = 3;
  il::SparseArray2C<double, int> A{};
  il::Array<double> y{};
  il::heat_3d(n, il::io, A, y);

  il::SparseArray2C<double, int> C = il::dot(il::io, A, A);

  return 0;
}

