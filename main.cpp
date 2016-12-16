//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <il/io/parse.h>
#include <string>

int main() {
  std::string text{"2.67e+5"};

  il::Status status{};
  double value = il::parse<double>(text, il::io, status);
  status.abort_on_error();
  IL_UNUSED(value);

  return 0;
}
