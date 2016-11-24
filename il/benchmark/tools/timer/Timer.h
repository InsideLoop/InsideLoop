//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_TIMER_H
#define IL_TIMER_H

#include <chrono>
#include <cstdint>

namespace il {

class Timer {
 private:
  double time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> point_begin_;

 public:
  Timer();
  void stop();
  double time() const;
};

Timer::Timer()
    : time_{0.0}, point_begin_{std::chrono::high_resolution_clock::now()} {}

void Timer::stop() {
  std::chrono::time_point<std::chrono::high_resolution_clock> point_end{
      std::chrono::high_resolution_clock::now()};
  time_ = 1.0e-9 *
          std::chrono::duration_cast<std::chrono::nanoseconds>(point_end -
                                                               point_begin_)
              .count();
}

double Timer::time() const { return time_; }

class TimerCycles {
 private:
  std::uint64_t point_begin_;
  std::uint64_t nb_cycles_;

 public:
  TimerCycles();
  void stop();
  long int cycles() const;
};

TimerCycles::TimerCycles() {
  std::uint32_t low;
  std::uint32_t high;
  asm volatile("rdtsc" : "=a"(low), "=d"(high));
  point_begin_ = static_cast<std::uint64_t>(low) |
                 (static_cast<std::uint64_t>(high) << 32);
}

void TimerCycles::stop() {
  std::uint32_t low;
  std::uint32_t high;
  asm volatile("rdtsc" : "=a"(low), "=d"(high));
  std::uint64_t point_end{static_cast<std::uint64_t>(low) |
                          (static_cast<std::uint64_t>(high) << 32)};
  nb_cycles_ = point_end - point_begin_;
}

long int TimerCycles::cycles() const {
  return static_cast<long int>(nb_cycles_);
}
}

#endif  // IL_TIMER_H
