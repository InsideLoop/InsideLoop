//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/String.h>

int main() {
  il::String prenom = "François";
  il::String nom = il::join(prenom, " Fayard");

  return 0;
}
