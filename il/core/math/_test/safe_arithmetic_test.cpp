//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/core/math/safe_arithmetic.h>

TEST(safe_arithmetic, sum_int_0) {
  const int a = std::numeric_limits<int>::max();
  const int b = 1;
  bool error = false;

  const int sum = il::safe_sum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, sum_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = -1;
  bool error = false;

  const int sum = il::safe_sum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, sum_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safe_sum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, sum_int_3) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safe_sum(a, b, il::io, error);
  IL_UNUSED(sum);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_0) {
  const int a = std::numeric_limits<int>::max();
  const int b = -1;
  bool error = false;

  const int difference = il::safe_difference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = 1;
  bool error = false;

  const int difference = il::safe_difference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::min();
  bool error = false;

  const int difference = il::safe_difference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_3) {
  const int a = std::numeric_limits<int>::min();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int difference = il::safe_difference(a, b, il::io, error);
  IL_UNUSED(difference);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_0) {
  const int a = std::numeric_limits<int>::max() / 2 + 1;
  const int b = 2;
  bool error = false;

  const int product = il::safe_product(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_1) {
  const int a = std::numeric_limits<int>::min() / 2 - 1;
  const int b = 2;
  bool error = false;

  const int product = il::safe_product(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int product = il::safe_product(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, product_int_3) {
  const int a = std::numeric_limits<int>::min();
  const int b = std::numeric_limits<int>::min();
  bool error = false;

  const int product = il::safe_product(a, b, il::io, error);
  IL_UNUSED(product);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, division_int_0) {
  const int a = 1;
  const int b = 0;
  bool error = false;

  const int quotient = il::safe_division(a, b, il::io, error);
  IL_UNUSED(quotient);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, division_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = -1;
  bool error = false;

  const int quotient = il::safe_division(a, b, il::io, error);
  IL_UNUSED(quotient);

  ASSERT_TRUE(error);
}


TEST(safe_arithmetic, safe_convert_0) {
  std::size_t a = std::numeric_limits<il::int_t>::max();
  a = a + 1;

  bool error = false;
  const il::int_t b = il::safe_convert<il::int_t>(a, il::io, error);
  IL_UNUSED(b);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, safe_convert_1) {
  const il::int_t a = -1;

  bool error = false;
  const std::size_t b = il::safe_convert<std::size_t>(a, il::io, error);
  IL_UNUSED(b);

  ASSERT_TRUE(error);
}
