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
  il::String filename = "/Users/fayard/Desktop/config.toml";

  il::Status status{};
  auto config =
      il::load<il::HashMap<il::String, il::Dynamic>>(filename, il::io, status);
  status.abort_on_error();

  // get the name
  il::String name{};
  il::int_t i = config.search("name");
  if (config.found(i) && config.value(i).is_string()) {
    name = config.value(i).as_string();
  }

  // get the number of cells
  il::int_t nb_cells;
  i = config.search("nb_cells");
  if (config.found(i) && config.value(i).is_integer()) {
    nb_cells = config.value(i).to_integer();
  }

  // get the property of the water
  double density;
  double compressibility;
  i = config.search("water");
  if (config.found(i) && config.value(i).is_hashmap()) {
    const il::HashMap<il::String, il::Dynamic>& water =
        config.value(i).as_const_hashmap();

    il::int_t j = water.search("density");
    if (water.found(j) && water.value(j).is_floating_point()) {
      density = water.value(j).to_floating_point();
    }

    j = water.search("compressibility");
    if (water.found(j) && water.value(j).is_floating_point()) {
      compressibility = water.value(j).to_floating_point();
    }
  }

  return 0;
}

