//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <il/Map.h>

TEST(Map, constructor_default_0) {
  il::Map<int, int> map{};

  ASSERT_TRUE(map.nbElements() == 0 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 0);
}

TEST(Map, constructor_initializer_list_0) {
  il::Map<int, int> map{il::value, {{0, 0}}};

  ASSERT_TRUE(map.nbElements() == 1 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 2);
}

TEST(Map, constructor_initializer_list_1) {
  il::Map<int, int> map{il::value, {{0, 0}, {1, 1}}};

  ASSERT_TRUE(map.nbElements() == 2 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 4);
}

TEST(Map, constructor_initializer_list_3) {
  il::Map<int, int> map{il::value, {{0, 0}, {1, 1}, {2, 2}}};

  ASSERT_TRUE(map.nbElements() == 3 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 4);
}

TEST(Map, constructor_initializer_list_4) {
  il::Map<int, int> map{il::value, {{0, 0}, {1, 1}, {2, 2}, {3, 3}}};

  ASSERT_TRUE(map.nbElements() == 4 && map.nbTombstones() == 0 &&
              map.nbBuckets() == 8);
}
