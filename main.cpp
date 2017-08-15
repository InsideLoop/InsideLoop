//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/Set.h>
#include <il/String.h>
#include <il/format.h>

int main() {
  il::String nom = "François";
  std::cout << il::format("Hello {0}! I wish a happy {1} for all {0}.", nom,
                          2017)
            << std::endl;

  il::Set<il::String> set{};
  set.insert("Fayard");
  set.insert("Bron");
  set.insert("François");
  set.insert("Frédéric");

  for (il::int_t i = set.first(); i != set.sentinel(); i = set.next(i)) {
    std::cout << set[i] << std::endl;
  }

  return 0;
}
