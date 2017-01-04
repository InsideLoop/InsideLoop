//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstdio>
#include <unordered_map>

#include <il/HashMap.h>
#include <il/Timer.h>

int main() {
  il::int_t n = 1000000;
  const il::int_t fac = 10000;
  il::Timer timer{};
  double x;

  timer.start();
  il::HashMap<il::int_t, il::int_t> map{n};
  x = 0.1;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = static_cast<il::int_t>(fac * n * x);
    il::int_t i = map.search(key);
    if (!map.found(i)) {
      map.insert(key, k, il::io, i);
    } else {
      map.value(i) = k;
    }
    x = 4 * x * (1 - x);
  }
  timer.stop();

  std::printf("      Time insert IL: %7.3f\n", timer.elapsed());
  std::printf("      Load factor IL: %7.3f\n", map.load());
  std::printf(" Displaced factor IL: %7.3f\n", map.displaced());
  std::printf("DDisplaced factor IL: %7.3f\n", map.displaced_twice());


  timer.reset();
  timer.start();
  il::int_t sum = 0;
  x = 0.1;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = static_cast<il::int_t>(fac * n * x);
    const il::int_t i = map.search(key);
    sum += map.value(i);
    x = 4 * x * (1 - x);
  }
  timer.stop();

  std::printf("      Time search IL: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  std::unordered_map<il::int_t, il::int_t> map_stl{};
  x = 0.1;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = static_cast<il::int_t>(fac * n * x);
    map_stl[key] = k;
    x = 4 * x * (1 - x);
  }
  timer.stop();

  std::printf("     Time insert STL: %7.3f\n", timer.elapsed());

  timer.reset();
  timer.start();
  il::int_t sum_stl = 0;
  x = 0.1;
  for (il::int_t k = 0; k < n; ++k) {
    const il::int_t key = static_cast<il::int_t>(fac * n * x);
    sum_stl += map_stl[key];
    x = 4 * x * (1 - x);
  }
  timer.stop();

  std::printf("     Time search STL: %7.3f\n", timer.elapsed());
  std::printf("Sum: %d %d\n", static_cast<int>(sum), static_cast<int>(sum_stl));

  return 0;
}
