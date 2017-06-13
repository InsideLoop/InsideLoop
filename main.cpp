//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/data.h>
#include <il/toml.h>

int main() {
  il::Status status{};

  il::HashMapArray<il::String, il::Dynamic> config{};
  config.set("First_name", "François");
  config.set("Last_name", "Fayard");

  il::String filename_toml = "/Users/fayard/Desktop/config.toml";
  il::save(config, filename_toml, il::io, status);
  status.abort_on_error();

  il::String filename_binary = "/Users/fayard/Desktop/config.data";
  il::save(config, filename_binary, il::io, status);
  status.abort_on_error();

  auto config_load_toml = il::load<il::HashMapArray<il::String, il::Dynamic>>(
      filename_toml, il::io, status);
  status.abort_on_error();

  auto config_load_data = il::load<il::HashMap<il::String, il::Dynamic>>(
      filename_binary, il::io, status);
  status.abort_on_error();

  return 0;
}
