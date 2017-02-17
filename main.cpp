//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/String.h>
#include <il/Dynamic.h>
#include <il/HashMap.h>

int main() {
  il::HashMap<il::String, il::Dynamic> car{};
  car.set("Make", "Renault");
  car.set("Model", "Clio");
  car.set("Seats", 4);
  car.set("Automatic", false);
  car.set("Speed", 165.1);

  for (il::int_t i = car.first(); i != car.last() ; i = car.next(i)) {
    std::cout << car.key(i).c_string() << std::endl;
  }

  return 0;
}
