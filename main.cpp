//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <bitset>
#include <iostream>

#include <il/Array.h>
#include <il/Dynamic.h>
#include <il/HashMapArray.h>
#include <il/String.h>
#include <il/Timer.h>
#include <il/io/filepack/filepack.h>
#include <il/string_util.h>

int main() {
  il::String filename_filepack = "/Users/fayard/Desktop/config.fp";

  il::Timer timer{};
  il::Status status{};

  timer.start();
  auto config_bis = il::load_filepack(filename_filepack, il::io, status);
  timer.stop();
  status.abort_on_error();

  std::cout << "Time to load: " << timer.elapsed() << " s" << std::endl;

  return 0;
}
