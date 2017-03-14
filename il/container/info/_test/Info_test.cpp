//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/Info.h>

TEST(Info, integer) {
  const il::int_t value = 34;

  il::Info info{};
  info.set("line", value);

  il::int_t line = value + 1;
  const il::int_t i = info.search("line");
  if (info.found(i) && info.is_integer(i)) {
    line = info.to_integer(i);
  }

  ASSERT_TRUE(line == value);
}

TEST(Info, double) {
  const double value = 3.14159;

  il::Info info{};
  info.set("pi", value);

  double pi = value + 1.0;
  const il::int_t i = info.search("pi");
  if (info.found(i) && info.is_double(i)) {
    pi = info.to_double(i);
  }

  ASSERT_TRUE(pi == value);
}
