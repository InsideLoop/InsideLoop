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

  ASSERT_TRUE(a.is_null() && !a.is_bool() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::null_t);
}

TEST(Dynamic, boolean_constructor_0) {
  il::Dynamic a = true;

  ASSERT_TRUE(a.is_bool() && a.to_bool() && !a.is_null() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::bool_t);
}

TEST(Dynamic, boolean_constructor_1) {
  il::Dynamic a = false;

  ASSERT_TRUE(a.is_bool() && a.to_bool() == false && !a.is_null() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::bool_t);
}

TEST(Dynamic, integer_constructor_0) {
  il::Dynamic a = 3;

  ASSERT_TRUE(a.is_integer() && a.to_integer() == 3 && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t);
}

TEST(Dynamic, integer_constructor_1) {
  il::Dynamic a = -3;

  ASSERT_TRUE(a.is_integer() && a.to_integer() == -3 && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t);
}

TEST(Dynamic, integer_constructor_2) {
  const il::int_t n = (il::int_t{1} << 47) - 1;
  il::Dynamic a = n;

  ASSERT_TRUE(a.is_integer() && a.to_integer() == n && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t);
}

TEST(Dynamic, integer_constructor_3) {
  const il::int_t n = -(il::int_t{1} << 47);
  il::Dynamic a = n;

  ASSERT_TRUE(a.is_integer() && a.to_integer() == n && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t);
}

TEST(Dynamic, double_constructor_0) {
  const double x = 3.14159;
  il::Dynamic a = x;

  bool b0 = a.is_double();
  IL_UNUSED(b0);

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, double_constructor_1) {
  const double x = 0.0 / 0.0;
  il::Dynamic a = x;

  ASSERT_TRUE(a.is_double() && std::isnan(a.to_double()) && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, double_constructor_2) {
  const double x = std::numeric_limits<double>::quiet_NaN();
  il::Dynamic a = x;

  ASSERT_TRUE(a.is_double() && std::isnan(a.to_double()) && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, double_constructor_3) {
  const double x = std::numeric_limits<double>::signaling_NaN();
  il::Dynamic a = x;

  ASSERT_TRUE(a.is_double() && std::isnan(a.to_double()) && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, double_constructor_4) {
  double x = std::numeric_limits<double>::signaling_NaN();
  x += 1.0;
  il::Dynamic a = x;

  ASSERT_TRUE(a.is_double() && std::isnan(a.to_double()) && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, double_constructor_5) {
  const double x = std::numeric_limits<double>::infinity();
  il::Dynamic a = x;

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, double_constructor_6) {
  const double x = -std::numeric_limits<double>::infinity();
  il::Dynamic a = x;

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, string_constructor_0) {
  const il::String string{};
  il::Dynamic a = string;

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t);
}

TEST(Dynamic, string_constructor_1) {
  const il::String string = "Hello";
  il::Dynamic a = string;

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t);
}

TEST(Dynamic, string_constructor_2) {
  const char* c_string = "Hello";
  const il::String string = c_string;
  il::Dynamic a = c_string;

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t);
}

TEST(Dynamic, copy_constructor_null) {
  il::Dynamic b{};
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_null() && !a.is_bool() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::null_t);
}

TEST(Dynamic, copy_constructor_boolean_0) {
  il::Dynamic b = true;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_bool() && a.to_bool() && !a.is_null() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::bool_t);
}

TEST(Dynamic, copy_constructor_boolean_1) {
  il::Dynamic b = false;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_bool() && a.to_bool() == false && !a.is_null() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::bool_t);
}

TEST(Dynamic, copy_constructor_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_integer() && a.to_integer() == 3 && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t);
}

TEST(Dynamic, copy_constructor_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, copy_constructor_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t);
}

TEST(Dynamic, copy_constructor_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_array() && a.as_const_array().size() == 3 &&
              a.as_const_array().capacity() == 3 &&
              a.as_const_array()[0].is_null() &&
              a.as_const_array()[1].is_null() &&
              a.as_const_array()[2].is_null() && !a.is_null() && !a.is_bool() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_hash_map_array() && a.type() == il::Type::array_t);
}

TEST(Dynamic, copy_constructor_hashmaparray_0) {
  il::HashMapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{b};

  ASSERT_TRUE(a.is_hash_map_array() && a.as_hash_map_array().size() == 2 &&
              a.as_hash_map_array().search("Hello") >= 0 &&
              a.as_hash_map_array().search("World!") >= 0 && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_string() && !a.is_array() &&
              a.type() == il::Type::hash_map_array_t);
}

TEST(Dynamic, move_constructor_null) {
  il::Dynamic b{};
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_null() && !a.is_bool() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::null_t &&
              b.is_null());
}

TEST(Dynamic, move_constructor_boolean_0) {
  il::Dynamic b = true;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_bool() && a.to_bool() && !a.is_null() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::bool_t &&
              b.is_null());
}

TEST(Dynamic, move_constructor_boolean_1) {
  il::Dynamic b = false;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_bool() && a.to_bool() == false && !a.is_null() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::bool_t && b.is_null());
}

TEST(Dynamic, move_constructor_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_integer() && a.to_integer() == 3 && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t && b.is_null());
}

TEST(Dynamic, move_constructor_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t && b.is_null());
}

TEST(Dynamic, move_constructor_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t && b.is_null());
}

TEST(Dynamic, move_constructor_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(
      a.is_array() && a.as_const_array().size() == 3 &&
      a.as_const_array().capacity() == 3 && a.as_const_array()[0].is_null() &&
      a.as_const_array()[1].is_null() && a.as_const_array()[2].is_null() &&
      !a.is_null() && !a.is_bool() && !a.is_integer() && !a.is_double() &&
      !a.is_string() && !a.is_hash_map_array() &&
      a.type() == il::Type::array_t && b.is_null());
}

TEST(Dynamic, move_constructor_hashmaparray_0) {
  il::HashMapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a = std::move(b);

  ASSERT_TRUE(a.is_hash_map_array() && a.as_hash_map_array().size() == 2 &&
              a.as_hash_map_array().search("Hello") >= 0 &&
              a.as_hash_map_array().search("World!") >= 0 && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_string() && !a.is_array() &&
              a.type() == il::Type::hash_map_array_t && b.is_null());
}

TEST(Dynamic, copy_assignement_null) {
  il::Dynamic b{};
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_null() && !a.is_bool() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::null_t);
}

TEST(Dynamic, copy_assignement_boolean_0) {
  il::Dynamic b = true;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_bool() && a.to_bool() && !a.is_null() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::bool_t);
}

TEST(Dynamic, copy_assignement_boolean_1) {
  il::Dynamic b = false;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_bool() && a.to_bool() == false && !a.is_null() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::bool_t);
}

TEST(Dynamic, copy_assignement_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_integer() && a.to_integer() == 3 && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t);
}

TEST(Dynamic, copy_assignement_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t);
}

TEST(Dynamic, copy_assignement_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t);
}

TEST(Dynamic, copy_assignement_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_array() && a.as_const_array().size() == 3 &&
              a.as_const_array().capacity() == 3 &&
              a.as_const_array()[0].is_null() &&
              a.as_const_array()[1].is_null() &&
              a.as_const_array()[2].is_null() && !a.is_null() && !a.is_bool() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_hash_map_array() && a.type() == il::Type::array_t);
}

TEST(Dynamic, copy_assignement_hashmaparray_0) {
  il::HashMapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{};
  a = b;

  ASSERT_TRUE(a.is_hash_map_array() && a.as_hash_map_array().size() == 2 &&
              a.as_hash_map_array().search("Hello") >= 0 &&
              a.as_hash_map_array().search("World!") >= 0 && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_string() && !a.is_array() &&
              a.type() == il::Type::hash_map_array_t);
}

TEST(Dynamic, move_assignement_null) {
  il::Dynamic b{};
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_null() && !a.is_bool() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::null_t &&
              b.is_null());
}

TEST(Dynamic, move_assignement_boolean_0) {
  il::Dynamic b = true;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_bool() && a.to_bool() && !a.is_null() && !a.is_integer() &&
              !a.is_double() && !a.is_string() && !a.is_array() &&
              !a.is_hash_map_array() && a.type() == il::Type::bool_t &&
              b.is_null());
}

TEST(Dynamic, move_assignement_boolean_1) {
  il::Dynamic b = false;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_bool() && a.to_bool() == false && !a.is_null() &&
              !a.is_integer() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::bool_t && b.is_null());
}

TEST(Dynamic, move_assignement_integer_0) {
  const il::int_t n = 3;
  il::Dynamic b = n;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_integer() && a.to_integer() == 3 && !a.is_null() &&
              !a.is_bool() && !a.is_double() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::integer_t && b.is_null());
}

TEST(Dynamic, move_assignement_floating_point_0) {
  const double x = 3.14159;
  il::Dynamic b = x;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_double() && a.to_double() == x && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_string() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::double_t && b.is_null());
}

TEST(Dynamic, move_assignement_string_0) {
  const il::String string = "Hello";
  il::Dynamic b = string;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_string() && a.as_string() == string && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_array() && !a.is_hash_map_array() &&
              a.type() == il::Type::string_t && b.is_null());
}

TEST(Dynamic, move_assignement_array_0) {
  const il::int_t n = 3;
  const il::Array<il::Dynamic> v{n};
  il::Dynamic b = v;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(
      a.is_array() && a.as_const_array().size() == 3 &&
      a.as_const_array().capacity() == 3 && a.as_const_array()[0].is_null() &&
      a.as_const_array()[1].is_null() && a.as_const_array()[2].is_null() &&
      !a.is_null() && !a.is_bool() && !a.is_integer() && !a.is_double() &&
      !a.is_string() && !a.is_hash_map_array() &&
      a.type() == il::Type::array_t && b.is_null());
}

TEST(Dynamic, move_assignement_hashmaparray_0) {
  il::HashMapArray<il::String, il::Dynamic> map{};
  map.set(il::String{"Hello"}, il::Dynamic{5});
  map.set(il::String{"World!"}, il::Dynamic{6});

  il::Dynamic b = map;
  il::Dynamic a{};
  a = std::move(b);

  ASSERT_TRUE(a.is_hash_map_array() && a.as_hash_map_array().size() == 2 &&
              a.as_hash_map_array().search("Hello") >= 0 &&
              a.as_hash_map_array().search("World!") >= 0 && !a.is_null() &&
              !a.is_bool() && !a.is_integer() && !a.is_double() &&
              !a.is_string() && !a.is_array() &&
              a.type() == il::Type::hash_map_array_t && b.is_null());
}
