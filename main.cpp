//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <iostream>
#include <string>

#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Array3D.h>
#include <il/HashTable.h>

int main() {
  il::Array2D<double> A{2, 3};
  il::Array3D<double> B{2, 3, 4};

  il::HashTable<std::string, int> table{};
  il::Error error{};
  table.insert("Marseille", 855393, il::io, error);
  table.insert("Paris", 2229621, il::io, error);
  table.insert("Lyon", 500715, il::io, error);
  error.ignore();

  for (auto it = table.begin(); it != table.end(); ++it) {
    std::cout << "La ville de " << it->key << " a " << it->value
              << " habitants.\n";
  }

  return 0;
}
