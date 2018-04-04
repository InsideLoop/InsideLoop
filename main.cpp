//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
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

#include <cmath>
#include <iostream>

#include <il/Array.h>
#include <il/Timer.h>

int main() {
  const il::int_t n = 10000000;
  il::Array<double> v{n};
  const double alpha = 1.0;

  il::Timer timer{};
  timer.Start();
  for (il::int_t k = 0; k < v.size(); ++k) {
    v[k] = std::cos(k * alpha);
  }
  timer.Stop();

  std::cout << "Time: " << timer.time() << "s" << std::endl;

  return 0;
}
