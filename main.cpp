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
  il::Toml config{};
  config.set("title", "drop");
  config.set("width", 640);
  config.set("height", 480);
  config.set("rho", 1.2345);

  il::String filename = "/home/fayard/Desktop/config.toml";

  il::Status status{};
  il::save(config, filename, il::io, status);
  status.abort_on_error();

  return 0;
}
