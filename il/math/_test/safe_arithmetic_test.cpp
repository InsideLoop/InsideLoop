//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/math/safe_arithmetic.h>

TEST(safe_arithmetic, addition_int_0) {
  const int a = std::numeric_limits<int>::max();
  const int b = 1;
  bool error = false;

  const int sum = il::safe_addition(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, addition_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = -1;
  bool error = false;

  const int sum = il::safe_addition(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, addition_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safe_addition(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, addition_int_3) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safe_addition(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_0) {
  const int a = std::numeric_limits<int>::max();
  const int b = -1;
  bool error = false;

  const int sum = il::safe_substraction(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = 1;
  bool error = false;

  const int sum = il::safe_substraction(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::min();
  bool error = false;

  const int sum = il::safe_substraction(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, substraction_int_3) {
  const int a = std::numeric_limits<int>::min();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safe_substraction(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, multiplication_int_0) {
  const int a = std::numeric_limits<int>::max() / 2 + 1;
  const int b = 2;
  bool error = false;

  const int sum = il::safe_multiplication(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, multiplication_int_1) {
  const int a = std::numeric_limits<int>::min() / 2 - 1;
  const int b = 2;
  bool error = false;

  const int sum = il::safe_multiplication(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, multiplication_int_2) {
  const int a = std::numeric_limits<int>::max();
  const int b = std::numeric_limits<int>::max();
  bool error = false;

  const int sum = il::safe_multiplication(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, multiplication_int_3) {
  const int a = std::numeric_limits<int>::min();
  const int b = std::numeric_limits<int>::min();
  bool error = false;

  const int sum = il::safe_multiplication(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, division_int_0) {
  const int a = 1;
  const int b = 0;
  bool error = false;

  const int sum = il::safe_division(a, b, il::io, error);

  ASSERT_TRUE(error);
}

TEST(safe_arithmetic, division_int_1) {
  const int a = std::numeric_limits<int>::min();
  const int b = -1;
  bool error = false;

  const int sum = il::safe_division(a, b, il::io, error);

  ASSERT_TRUE(error);
}
