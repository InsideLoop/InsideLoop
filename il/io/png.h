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

#include <il/Array3D.h>
#include <il/core/Status.h>

namespace il {

struct png_t {};
const png_t png{};

il::Array3D<unsigned char> load(const std::string& filename, il::png_t,
                                il::io_t, il::Status &status);

void save(const il::Array3D<unsigned char>& v, const std::string& filename,
          il::png_t, il::io_t, il::Status &status);
}

#endif  // IL_PNG_H
