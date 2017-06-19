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

TEST(String, default_constructor) {
  il::String s{};
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 0 && s.small() && s.capacity() == 23 && p[0] == '\0');
}

TEST(String, c_string_constructor_0) {
  il::String s{"A quite small string !!"};
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 23 && s.small() && s.capacity() == 23 &&
              0 == std::strcmp(p, "A quite small string !!"));
}

TEST(String, c_string_constructor_1) {
  il::String s{"A quite large string !!!"};
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 24 && !s.small() &&
              0 == std::strcmp(p, "A quite large string !!!"));
}

TEST(String, c_string_constructor_2) {
  il::String s{"A quite\0 large string !!!"};
  const char* p = s.asCString();

  ASSERT_TRUE(s.size() == 7 && s.small() && 0 == std::strcmp(p, "A quite"));
}

TEST(String, reserve_0) {
  il::String s{"A quite small string !!"};

  s.reserve(23);
  ASSERT_TRUE(s.size() == 23 && s.small() && s.capacity() == 23 &&
              0 == std::strcmp(s.asCString(), "A quite small string !!"));
}

TEST(String, reserve_1) {
  il::String s{"A quite small string !!"};

  s.reserve(24);
  ASSERT_TRUE(s.size() == 23 && !s.small() && s.capacity() == 24 &&
              0 == std::strcmp(s.asCString(), "A quite small string !!"));
}

TEST(String, reserve_2) {
  il::String s{"A quite large string !!!"};

  s.reserve(30);
  ASSERT_TRUE(s.size() == 24 && !s.small() && s.capacity() == 30 &&
              0 == std::strcmp(s.asCString(), "A quite large string !!!"));
}

TEST(String, append_0) {
  il::String s{"Hello"};
  s.append(" world!");

  ASSERT_TRUE(s.size() == 12 && s.small() && s.capacity() == 23 &&
              0 == std::strcmp(s.asCString(), "Hello world!"));
}

TEST(String, append_1) {
  il::String s{"Hello"};
  s.append(" world! I am so happy to be there");

  ASSERT_TRUE(s.size() == 38 && !s.small() && s.capacity() >= 38 &&
              0 == std::strcmp(s.asCString(),
                               "Hello world! I am so happy to be there"));
}

TEST(String, append_2) {
  il::String s{"Hello"};
  s.reserve(38);
  const char* p_before = s.asCString();
  s.append(" world! I am so happy to be there");
  const char* p_after = s.asCString();

  ASSERT_TRUE(s.size() == 38 && !s.small() && s.capacity() >= 38 &&
              p_before == p_after &&
              0 == std::strcmp(s.asCString(),
                               "Hello world! I am so happy to be there"));
}

TEST(String, append_3) {
  il::String s{"Hello world! I am so happy to be "};
  s.append("there");

  ASSERT_TRUE(s.size() == 38 && !s.small() && s.capacity() >= 38 &&
              0 == std::strcmp(s.asCString(),
                               "Hello world! I am so happy to be there"));
}

TEST(String, append_4) {
  il::String s = "HelloHelloHelloHello";
  s.append(s);

  ASSERT_TRUE(s.size() == 40 &&
              s == "HelloHelloHelloHelloHelloHelloHelloHello");
}
