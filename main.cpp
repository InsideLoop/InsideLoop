//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#include <cmath>
#include <iostream>

#include <il/Array3C.h>
#include <il/io/png/png.h>
#include <il/math.h>

int main() {
  const il::int_t n = 10000;
  const il::int_t half = n / 2;
  const double dx = 1.0 / half;
  const double dy = 1.0 / half;
  const double dtheta = 2 * il::pi / (2 * 360);
  const double r_inner = 0.05;
  const double r_inside = 0.1;
  const double r_outside = 1.0;
  const double width_circle = 0.02;

  il::Array3C<unsigned char> image{n, n, 3, 0};
  for (il::int_t ky = 0; ky < n; ++ky) {
    for (il::int_t kx = 0; kx < n; ++kx) {
      const double x = dx * (kx - half);
      const double y = dy * (ky - half);
      const double d = std::sqrt(x * x + y * y);
      const double theta = 2 * std::atan(y / (d + x));
      const il::int_t ktheta = il::floor(theta / dtheta);
      if (d <= r_inner) {
        image(ky, kx, 0) = 0;
        image(ky, kx, 1) = 0;
        image(ky, kx, 2) = 0;
      } else if (d >= r_inside && d <= r_outside) {
        const double x = 10 * (d - r_inside) / (r_outside - r_inside);

        if (il::abs(x - static_cast<il::int_t>(x)) <= width_circle ||
            il::abs(x - static_cast<il::int_t>(x)) >= 1.0 - width_circle) {
          image(ky, kx, 0) = 0;
          image(ky, kx, 1) = 0;
          image(ky, kx, 2) = 0;
        } else {
          image(ky, kx, 0) = (ktheta % 2 == 0) ? 0 : 255;
          image(ky, kx, 1) = (ktheta % 2 == 0) ? 0 : 255;
          image(ky, kx, 2) = (ktheta % 2 == 0) ? 0 : 255;
        }
      } else {
        image(ky, kx, 0) = 255;
        image(ky, kx, 1) = 255;
        image(ky, kx, 2) = 255;
      }
    }
  }

  const il::String filename = "/Users/fayard/Desktop/mire.png";
  il::Status status{};
  il::savePng(image, filename, il::io, status);
  status.abortOnError();

  return 0;
}
