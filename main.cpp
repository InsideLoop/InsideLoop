//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cmath>
#include <cstdio>

#include <il/Array3D.h>
#include <il/io/png.h>
#include <il/random/sobol.h>

int main() {
  const il::int_t n = 500;
  const il::int_t nb_point = 100000;
  const il::int_t dim = 2;

  il::Array2C<double> A = il::sobol(nb_point, dim, 0.0, 1.0);

  il::Array3D<unsigned char> image{n, n, 3, 255};
  for (il::int_t i = 0; i < nb_point; ++i) {
    il::int_t kx = std::floor(n * A(i, 0));
    il::int_t ky = std::floor(n * A(i, 1));
    image(kx, ky, 0) = 0;
    image(kx, ky, 1) = 0;
    image(kx, ky, 2) = 0;
  }

  il::Status status{};
  il::save(image, std::string{"/home/fayard/Desktop/sobol.png"}, il::png,
           il::io, status);
  status.ignore_error();

  return 0;
}
