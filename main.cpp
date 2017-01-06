//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <random>

#include <il/HashMap.h>
#include <il/Timer.h>
#include <il/format.h>

int main() {
  il::int_t n = 10;

  std::default_random_engine engine{};
  std::uniform_int_distribution<il::int_t> dist{
      0, std::numeric_limits<il::int_t>::max() / 2};
  il::Array<il::int_t> v{n};
  for (il::int_t k = 0; k < n; ++k) {
    v[k] = dist(engine);
  }

  il::Timer timer{};
  timer.start();
  il::HashMap<il::int_t, il::int_t> map{n};
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    il::int_t i = map.search(key);
    if (!map.found(i)) {
      map.insert(key, k, il::io, i);
    } else {
      map.value(i) = k;
    }
  }
  timer.stop();

  il::print("{:>15}: {:>10} s\n", "Time insert IL", timer.elapsed());

  timer.reset();
  timer.start();
  il::int_t sum = 0;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = v[k];
    const il::int_t i = map.search(key);
    sum += map.value(i);
  }
  timer.stop();

  il::print("{:>15}: {:>10} s\n", "Time search IL", timer.elapsed());
  il::print("{:>15}: {:>10}\n", "Sum", static_cast<int>(sum));

  return 0;
}
