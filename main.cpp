//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#include <iostream>

#include <il/Array.h>
#include <il/algorithmArray.h>
#include <il/Timer.h>

int main() {
//  const il::int_t n = 100000000;
  const il::int_t n = 10;
  il::Array<double> v{n};
  double x = 0.12345678987654321;
  for (il::int_t i = 0; i < n; ++i) {
    v[i] = x;
    x = 4 * x * (1 - x);
  }

  il::Timer timer{};

  timer.reset();
  {
    il::Array<double> w = v;
    timer.start();
    std::sort(w.begin(), w.end());
    timer.stop();
  }

  std::cout << "std: " << timer.time() << " s" << std::endl;

  timer.reset();
  {
    il::Array<double> w = v;
    timer.start();
    il::sort(il::io, w);
    timer.stop();
  }

  std::cout << "il: " << timer.time() << " s" << std::endl;

  return 0;
}
