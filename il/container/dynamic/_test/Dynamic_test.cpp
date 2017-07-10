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

#include <il/Dynamic.h>

TEST(Dynamic, implementation) {
  ASSERT_TRUE(sizeof(double) == 8 && sizeof(void*) == 8 &&
              sizeof(unsigned short) == 2 && sizeof(int) == 4 &&
              sizeof(long long) == 8 && sizeof(il::int_t) == 8);
}

TEST(Dynamic, default_constructor) {
  il::Dynamic a{};

  ASSERT_TRUE(a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isArray() && !a.isMapArray() &&
              a.type() == il::Type::Null);
}

TEST(Dynamic, bool_constructor_0) {
  il::Dynamic a = true;

  ASSERT_TRUE(a.isBool() && a.toBool() && !a.isNull() && !a.isInteger() &&
              !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Bool);
}

TEST(Dynamic, bool_constructor_1) {
  il::Dynamic a = false;

  ASSERT_TRUE(a.isBool() && a.toBool() == false && !a.isNull() &&
              !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && !a.isMapArray() && a.type() == il::Type::Bool);
}

TEST(Dynamic, integer_constructor_0) {
  il::Dynamic a = 3;

  ASSERT_TRUE(a.isInteger() && a.toInteger() == 3 && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer);
}

TEST(Dynamic, integer_constructor_1) {
  il::Dynamic a = -3;

  ASSERT_TRUE(a.isInteger() && a.toInteger() == -3 && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer);
}

TEST(Dynamic, integer_constructor_2) {
  const il::int_t n = (il::int_t{1} << 47) - 1;
  il::Dynamic a = n;

  ASSERT_TRUE(a.isInteger() && a.toInteger() == n && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer);
}

TEST(Dynamic, integer_constructor_3) {
  const il::int_t n = -(il::int_t{1} << 47);
  il::Dynamic a = n;

  ASSERT_TRUE(a.isInteger() && a.toInteger() == n && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer);
}

TEST(Dynamic, double_constructor_0) {
  const double x = 3.14159;
  il::Dynamic a = x;

  bool b0 = a.isDouble();
  IL_UNUSED(b0);

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, double_constructor_1) {
  const double x = 0.0 / 0.0;
  il::Dynamic a = x;

  ASSERT_TRUE(a.isDouble() && std::isnan(a.toDouble()) && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, double_constructor_2) {
  const double x = std::numeric_limits<double>::quiet_NaN();
  il::Dynamic a = x;

  ASSERT_TRUE(a.isDouble() && std::isnan(a.toDouble()) && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, double_constructor_3) {
  const double x = std::numeric_limits<double>::signaling_NaN();
  il::Dynamic a = x;

  ASSERT_TRUE(a.isDouble() && std::isnan(a.toDouble()) && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, double_constructor_4) {
  double x = std::numeric_limits<double>::signaling_NaN();
  x += 1.0;
  il::Dynamic a = x;

  ASSERT_TRUE(a.isDouble() && std::isnan(a.toDouble()) && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, double_constructor_5) {
  const double x = std::numeric_limits<double>::infinity();
  il::Dynamic a = x;

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, double_constructor_6) {
  const double x = -std::numeric_limits<double>::infinity();
  il::Dynamic a = x;

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, string_constructor_0) {
  const il::String string{};
  il::Dynamic a = string;

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String);
}

TEST(Dynamic, string_constructor_1) {
  const il::String string = "Hello";
  il::Dynamic a = string;

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String);
}

TEST(Dynamic, string_constructor_2) {
  const char* c_string = "Hello";
  const il::String string = c_string;
  il::Dynamic a = c_string;

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String);
}

TEST(Dynamic, copy_constructor_null) {
  il::Dynamic b{};
  il::Dynamic a{b};

  ASSERT_TRUE(a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isArray() && !a.isMapArray() &&
              a.type() == il::Type::Null);
}

TEST(Dynamic, copy_constructor_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isBool() && a.toBool() && !a.isNull() && !a.isInteger() &&
              !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Bool);
}

TEST(Dynamic, copy_constructor_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isBool() && a.toBool() == false && !a.isNull() &&
              !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && !a.isMapArray() && a.type() == il::Type::Bool);
}

TEST(Dynamic, copy_constructor_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isInteger() && a.toInteger() == 3 && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer);
}

TEST(Dynamic, copy_constructor_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, copy_constructor_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String);
}

TEST(Dynamic, copy_constructor_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isArray() && a.asArray().size() == 3 &&
              a.asArray().capacity() == 3 && a.asArray()[0].isNull() &&
              a.asArray()[1].isNull() && a.asArray()[2].isNull() &&
              !a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isMapArray() && a.type() == il::Type::Array);
}

TEST(Dynamic, copy_constructor_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{b};

  ASSERT_TRUE(a.isMapArray() && a.asMapArray().size() == 2 &&
              a.asMapArray().search("Hello") >= 0 &&
              a.asMapArray().search("World!") >= 0 && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && a.type() == il::Type::MapArray);
}

TEST(Dynamic, move_constructor_null) {
  il::Dynamic b{};
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isArray() && !a.isMapArray() &&
              a.type() == il::Type::Null && b.isNull());
}

TEST(Dynamic, move_constructor_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isBool() && a.toBool() && !a.isNull() && !a.isInteger() &&
              !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Bool && b.isNull());
}

TEST(Dynamic, move_constructor_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isBool() && a.toBool() == false && !a.isNull() &&
              !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && !a.isMapArray() && a.type() == il::Type::Bool &&
              b.isNull());
}

TEST(Dynamic, move_constructor_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isInteger() && a.toInteger() == 3 && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer && b.isNull());
}

TEST(Dynamic, move_constructor_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double && b.isNull());
}

TEST(Dynamic, move_constructor_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String && b.isNull());
}

TEST(Dynamic, move_constructor_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isArray() && a.asArray().size() == 3 &&
              a.asArray().capacity() == 3 && a.asArray()[0].isNull() &&
              a.asArray()[1].isNull() && a.asArray()[2].isNull() &&
              !a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isMapArray() && a.type() == il::Type::Array &&
              b.isNull());
}

TEST(Dynamic, move_constructor_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.isMapArray() && a.asMapArray().size() == 2 &&
              a.asMapArray().search("Hello") >= 0 &&
              a.asMapArray().search("World!") >= 0 && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && a.type() == il::Type::MapArray && b.isNull());
}

TEST(Dynamic, copy_assignement_null) {
  il::Dynamic b{};
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isArray() && !a.isMapArray() &&
              a.type() == il::Type::Null);
}

TEST(Dynamic, copy_assignement_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isBool() && a.toBool() && !a.isNull() && !a.isInteger() &&
              !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Bool);
}

TEST(Dynamic, copy_assignement_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isBool() && a.toBool() == false && !a.isNull() &&
              !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && !a.isMapArray() && a.type() == il::Type::Bool);
}

TEST(Dynamic, copy_assignement_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isInteger() && a.toInteger() == 3 && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer);
}

TEST(Dynamic, copy_assignement_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double);
}

TEST(Dynamic, copy_assignement_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String);
}

TEST(Dynamic, copy_assignement_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isArray() && a.asArray().size() == 3 &&
              a.asArray().capacity() == 3 && a.asArray()[0].isNull() &&
              a.asArray()[1].isNull() && a.asArray()[2].isNull() &&
              !a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isMapArray() && a.type() == il::Type::Array);
}

TEST(Dynamic, copy_assignement_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.isMapArray() && a.asMapArray().size() == 2 &&
              a.asMapArray().search("Hello") >= 0 &&
              a.asMapArray().search("World!") >= 0 && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && a.type() == il::Type::MapArray);
}

TEST(Dynamic, move_assignement_null) {
  il::Dynamic b{};
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isArray() && !a.isMapArray() &&
              a.type() == il::Type::Null && b.isNull());
}

TEST(Dynamic, move_assignement_bool_0) {
  il::Dynamic b = true;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isBool() && a.toBool() && !a.isNull() && !a.isInteger() &&
              !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Bool && b.isNull());
}

TEST(Dynamic, move_assignement_bool_1) {
  il::Dynamic b = false;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isBool() && a.toBool() == false && !a.isNull() &&
              !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && !a.isMapArray() && a.type() == il::Type::Bool &&
              b.isNull());
}

TEST(Dynamic, move_assignement_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isInteger() && a.toInteger() == 3 && !a.isNull() &&
              !a.isBool() && !a.isDouble() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Integer && b.isNull());
}

TEST(Dynamic, move_assignement_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isDouble() && a.toDouble() == x && !a.isNull() && !a.isBool() &&
              !a.isInteger() && !a.isString() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::Double && b.isNull());
}

TEST(Dynamic, move_assignement_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isString() && a.asString() == string && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isArray() &&
              !a.isMapArray() && a.type() == il::Type::String && b.isNull());
}

TEST(Dynamic, move_assignement_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isArray() && a.asArray().size() == 3 &&
              a.asArray().capacity() == 3 && a.asArray()[0].isNull() &&
              a.asArray()[1].isNull() && a.asArray()[2].isNull() &&
              !a.isNull() && !a.isBool() && !a.isInteger() && !a.isDouble() &&
              !a.isString() && !a.isMapArray() && a.type() == il::Type::Array &&
              b.isNull());
}

TEST(Dynamic, move_assignement_hashmaparray_0) {
  il::MapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.isMapArray() && a.asMapArray().size() == 2 &&
              a.asMapArray().search("Hello") >= 0 &&
              a.asMapArray().search("World!") >= 0 && !a.isNull() &&
              !a.isBool() && !a.isInteger() && !a.isDouble() && !a.isString() &&
              !a.isArray() && a.type() == il::Type::MapArray && b.isNull());
}
