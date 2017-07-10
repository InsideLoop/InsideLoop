//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/String.h>
#include <il/StringView.h>

TEST(StringView, Rune_0) {
  il::String s = "abc";
  il::StringView sv{s.data(), s.size()};

  const il::int_t i0 = 0;
  const il::int_t i1 = sv.nextRune(i0);
  const il::int_t i2 = sv.nextRune(i1);
  const il::int_t i3 = sv.nextRune(i2);

  ASSERT_TRUE(s.size() == 3 && sv.toRune(i0) == 97 && sv.toRune(i1) == 98 &&
              sv.toRune(i2) == 99 && i3 == 3);
}

TEST(StringView, Rune_1) {
  il::String s = u8"nço";
  il::StringView sv{s.data(), s.size()};

  const il::int_t i0 = 0;
  const il::int_t i1 = sv.nextRune(i0);
  const il::int_t i2 = sv.nextRune(i1);
  const il::int_t i3 = sv.nextRune(i2);

  ASSERT_TRUE(s.size() == 4 && sv.toRune(i0) == 110 && sv.toRune(i1) == 231 &&
              sv.toRune(i2) == 111 && i3 == 4);
}
