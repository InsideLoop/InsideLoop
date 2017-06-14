//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/Array.h>
#include <il/String.h>

#include <il/unicode.h>
#include <il/numpy.h>

int main() {
  il::String filename = "/Users/fayard/Desktop/";
  filename.append(il::CodePoint::grinning_face_with_smiling_eyes);
  filename.append(".npy");

  il::Array<double> v{10};

  il::Status status{};
  il::save(v, filename, il::io, status);
  status.abort_on_error();

  return 0;
}
