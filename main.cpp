//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/Dynamic.h>
#include <il/HashMap.h>
#include <il/String.h>
#include <il/Toml.h>

int main() {
  il::String filename = "/Users/fayard/Desktop/config.toml";

  il::Status status{};
  auto toml = il::load<il::Toml>(filename, il::io, status);
  status.abort_on_error();

  il::int_t i = toml.search("Price");
  if (toml.found(i) && toml.value(i).is_hashmap()) {
    const il::Toml& engine = toml.value(i).as_const_hashmap();
    il::int_t j = engine.search("retail");
    if (engine.found(j) && engine.value(j).is_hashmap()) {
      const il::Toml& season = engine.value(j).as_const_hashmap();
      for (il::int_t k = season.first(); k != season.last();
           k = season.next(k)) {
        if (season.value(k).is_integer()) {
          std::cout << season.key(k).c_string() << ": "
                    << season.value(k).get_integer() << std::endl;
        }
      }
    }
  }

  return 0;
}
