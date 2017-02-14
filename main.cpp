//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/io/yaml.h>

int main() {
  il::String filename = "/home/fayard/Desktop/config.yaml";

  il::Status status{};
  il::Yaml config = il::load<il::Yaml>(filename, il::io, status);
  status.abort_on_error();

  return 0;
}
