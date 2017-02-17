//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/io/toml.h>

int main() {
  il::String filename_in = "/home/fayard/Desktop/config.toml";
  il::String filename_out = "/home/fayard/Desktop/config-out.toml";

  il::Status status{};
  auto toml = il::load<il::Toml>(filename_in, il::io, status);
  status.abort_on_error();

  il::save(toml, filename_out, il::io, status);
  status.abort_on_error();

  return 0;
}
