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
  il::Yaml config{};
  config.append("Permeability", 1.0e-15);
  config.append("Nodes", 100);
  config.append("Name", "Francois Fayard");

  il::Status status{};
  il::save(config, il::String{"/home/fayard/Desktop/config.yaml"}, il::io,
           status);
  status.abort_on_error();

  il::Yaml cfg = il::load<il::Yaml>(
      il::String{"/home/fayard/Desktop/config.yaml"}, il::io, status);
  status.abort_on_error();

  il::save(config, il::String{"/home/fayard/Desktop/cfg.yaml"}, il::io,
           status);
  status.abort_on_error();

  return 0;
}
