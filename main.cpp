//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>

#include <il/Map.h>

int main() {
  il::Map<int, int> map{};
  map.insert(0, 5);
  map.insert(1, 2);
  std::cout << map.nbElements() << " " << map.nbBuckets() << std::endl;

  map.clear();
  std::cout << map.nbElements() << " " << map.nbBuckets() << std::endl;

  return 0;
}
