//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_TIME_H
#define IL_TIME_H

#include <string>

#include <il/base.h>

namespace il {

const double nanosecond{1.0e-9};
const double microsecond{1.0e-6};
const double millisecond{1.0e-3};
const double second = 1.0;
const double minute = 60.0;
const double hour{60.0 * minute};
const double day{24.0 * hour};
const double year{365.0 * day};

inline std::string time_to_string(double time) {
  IL_EXPECT_FAST(time >= 0);

  if (time == 0.0) {
    return std::string{"0 second"};
  } else if (time < il::nanosecond) {
    return std::to_string(time) + std::string{" seconds"};
  } else if (time < il::microsecond) {
    return std::to_string(time / il::nanosecond) + std::string{" nanoseconds"};
  } else if (time < il::millisecond) {
    return std::to_string(time / il::microsecond) +
           std::string{" microseconds"};
  } else if (time < il::second) {
    return std::to_string(time / il::millisecond) +
           std::string{" milliseconds"};
  } else if (time < il::minute) {
    return std::to_string(time) + std::string{" seconds"};
  } else if (time < il::hour) {
    return std::to_string(time / il::minute) + std::string{" minutes"};
  } else if (time < il::day) {
    return std::to_string(time / il::hour) + std::string{" hours"};
  } else if (time < il::year) {
    return std::to_string(time / il::day) + std::string{" days"};
  } else {
    return std::to_string(time / il::year) + std::string{" years"};
  }
}
}  // namespace il

#endif  // IL_TIME_H
