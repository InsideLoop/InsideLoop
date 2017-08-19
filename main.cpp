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
#include <il/Map.h>

int main() {
  il::String s = "1234567890123456789012";
  il::String nom = "FrançoisFa";
  s.insert(5, nom);

  il::Map<il::String, double> map{};
  map.insertCString("e", 8.0);
  double g = map.valueForCString("g", 9.81);
  IL_UNUSED(g);

  return 0;
}
