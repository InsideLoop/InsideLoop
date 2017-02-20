//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Toml.h>

int main() {
  il::String filename = "/home/fayard/Desktop/config.toml";

  il::Status status{};
  auto toml =
      il::load<il::HashMap<il::String, il::Dynamic>>(filename, il::io, status);
  status.abort_on_error();

  return 0;
}
