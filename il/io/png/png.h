//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_PNG_H
#define IL_PNG_H

#include <il/Array2C.h>
#include <il/Array3C.h>
#include <il/Status.h>
#include <il/String.h>

#include <il/io/io_base.h>

namespace il {

il::Array3C<unsigned char> loadPng(const il::String& filename, il::io_t,
                                   il::Status& status);

void savePng(const il::Array3C<unsigned char>& v, const il::String& filename,
             il::io_t, il::Status& status);

template <>
class SaveHelperPng<il::Array3C<unsigned char>> {
 public:
  static void save(const il::Array3C<unsigned char>& image,
                   const il::String& filename, il::io_t, il::Status& status) {
    savePng(image, filename, il::io, status);
  }
};

template <>
class SaveHelperPng<il::Array2C<unsigned char>> {
 public:
  static void save(const il::Array2C<unsigned char>& image,
                   const il::String& filename, il::io_t, il::Status& status) {
    il::Array3C<unsigned char> color_image{image.size(0), image.size(1), 3};
    for (il::int_t ky = 0; ky < image.size(0); ++ky) {
      for (il::int_t kx = 0; kx < image.size(1); ++kx) {
        color_image(ky, kx, 0) = image(ky, kx);
        color_image(ky, kx, 1) = image(ky, kx);
        color_image(ky, kx, 2) = image(ky, kx);
      }
    }
    savePng(color_image, filename, il::io, status);
  }
};

}  // namespace il

#endif  // IL_PNG_H
