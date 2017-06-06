//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/Toml.h>

#include <iostream>

il::String directory =
    "/home/fayard/Documents/Projects/InsideLoop/InsideLoop/il/io/toml/_test/"
    "valid/";

TEST(Toml, array_empty) {
  bool ans = true;

  il::String filename = directory;
  filename.append("array-empty.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i0 = config.search("thevoid");
    if (!(config.found(i0) && config.value(i0).is_array())) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array1 = config.value(i0).as_array();
      if (array1.size() != 1 || !(array1[0].is_array())) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &array2 = array1[0].as_array();
        if (array2.size() != 1 || !(array2[0].is_array())) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &array3 = array2[0].as_array();
          if (array3.size() != 1 || !(array3[0].is_array())) {
            ans = false;
          } else {
            const il::Array<il::Dynamic> &array4 = array3[0].as_array();
            if (array4.size() != 1 || !(array4[0].is_array())) {
              ans = false;
            } else {
              const il::Array<il::Dynamic> &array5 = array4[0].as_array();
              if (array5.size() != 0) {
                ans = false;
              }
            }
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, array_nospaces) {
  bool ans = true;

  il::String filename = directory;
  filename.append("array-nospaces.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("ints");
    if (!(config.found(i) && config.value(i).is_array())) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array = config.value(i).as_array();
      if (array.size() != 3 || !array[0].is_integer() ||
          array[0].to_integer() != 1 || !array[1].is_integer() ||
          array[1].to_integer() != 2 || !array[2].is_integer() ||
          array[2].to_integer() != 3) {
        ans = false;
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, arrays_heterogeneous) {
  bool ans = true;

  il::String filename = directory;
  filename.append("arrays-hetergeneous.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("mixed");
    if (!(config.found(i) && config.value(i).is_array() &&
          config.value(i).as_array().size() == 3)) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array = config.value(i).as_array();

      if (!(array[0].is_array() && array[0].as_array().size() == 2)) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &subarray0 = array[0].as_array();
        if (!(subarray0[0].is_integer() && subarray0[0].to_integer() == 1 &&
              subarray0[1].is_integer() && subarray0[1].to_integer() == 2)) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &subarray1 = array[1].as_array();
          if (!(subarray1[0].is_string() && subarray1[0].as_string() == "a" &&
                subarray1[1].is_string() && subarray1[1].as_string() == "b")) {
            ans = false;
          } else {
            const il::Array<il::Dynamic> &subarray2 = array[2].as_array();
            if (!(subarray2[0].is_floating_point() &&
                  subarray2[0].to_floating_point() == 1.1 &&
                  subarray2[1].is_floating_point() &&
                  subarray2[1].to_floating_point() == 2.1)) {
              ans = false;
            }
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, arrays_nested) {
  bool ans = true;

  il::String filename = directory;
  filename.append("arrays-nested.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("nest");
    if (!(config.found(i) && config.value(i).is_array() &&
          config.value(i).as_array().size() == 2)) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array = config.value(i).as_array();

      if (!(array[0].is_array() && array[0].as_array().size() == 1 &&
            array[1].is_array() && array[1].as_array().size() == 1)) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &subarray0 = array[0].as_array();
        if (!(subarray0[0].is_string() && subarray0[0].as_string() == "a")) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &subarray1 = array[1].as_array();
          if (!(subarray1[0].is_string() && subarray1[0].as_string() == "b")) {
            ans = false;
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, boolean) {
  bool ans = true;

  il::String filename = directory;
  filename.append("bool.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 2) {
    ans = false;
  } else {
    const il::int_t i = config.search("t");
    if (!(config.found(i) && config.value(i).is_boolean() &&
          config.value(i).to_boolean())) {
      ans = false;
    } else {
      const il::int_t i = config.search("f");
      if (!(config.found(i) && config.value(i).is_boolean() &&
            config.value(i).to_boolean() == false)) {
        ans = false;
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, comments_everywhere) {
  bool ans = true;

  il::String filename = directory;
  filename.append("comments-everywhere.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("group");
    if (!(config.found(i) && config.value(i).is_hashmaparray())) {
      ans = false;
    } else {
      il::HashMapArray<il::String, il::Dynamic> &group =
          config.value(i).as_hashmaparray();
      il::int_t j0 = group.search("answer");
      il::int_t j1 = group.search("more");
      if (!(group.size() == 2 && group.found(j0) &&
            group.value(j0).is_integer() &&
            group.value(j0).to_integer() == 42 && group.found(j1) &&
            group.value(j1).is_array())) {
        ans = false;
      } else {
        il::Array<il::Dynamic> &array = group.value(j1).as_array();
        if (!(array.size() == 2 && array[0].is_integer() &&
              array[0].to_integer() == 42 && array[1].is_integer() &&
              array[1].to_integer() == 42)) {
          ans = false;
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, empty) {
  bool ans = true;

  il::String filename = directory;
  filename.append("empty.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 0) {
    ans = false;
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, double) {
  bool ans = true;

  il::String filename = directory;
  filename.append("float.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("pi");
    il::int_t i1 = config.search("negpi");
    if (!(config.found(i0) && config.value(i0).is_floating_point() &&
          config.value(i0).to_floating_point() == 3.14 && config.found(i1) &&
          config.value(i1).is_floating_point() &&
          config.value(i1).to_floating_point() == -3.14)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, implicit_and_explicit_after) {
  bool ans = true;

  il::String filename = directory;
  filename.append("implicit-and-explicit-after.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a");
    if (!(config.found(i) && config.value(i).is_hashmaparray())) {
      ans = false;
    } else {
      il::HashMapArray<il::String, il::Dynamic> &a =
          config.value(i).as_hashmaparray();
      il::int_t i0 = a.search("better");
      il::int_t i1 = a.search("b");
      if (!(a.size() == 2 && a.found(i0) && a.value(i0).is_integer() &&
            a.value(i0).to_integer() == 43 && a.found(i1) &&
            a.value(i1).is_hashmaparray())) {
        ans = false;
      } else {
        il::HashMapArray<il::String, il::Dynamic> &b =
            a.value(i1).as_hashmaparray();
        il::int_t j = b.search("c");
        if (!(b.size() == 1 && b.found(j) && b.value(j).is_hashmaparray())) {
          ans = false;
        } else {
          il::HashMapArray<il::String, il::Dynamic> &c =
              b.value(j).as_hashmaparray();
          il::int_t j0 = c.search("answer");
          if (!(c.size() == 1 && c.found(j0) && c.value(j0).is_integer() &&
                c.value(j0).to_integer() == 42)) {
            ans = false;
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, implicit_and_explicit_before) {
  bool ans = true;

  il::String filename = directory;
  filename.append("implicit-and-explicit-before.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a");
    if (!(config.found(i) && config.value(i).is_hashmaparray())) {
      ans = false;
    } else {
      il::HashMapArray<il::String, il::Dynamic> &a =
          config.value(i).as_hashmaparray();
      il::int_t i0 = a.search("better");
      il::int_t i1 = a.search("b");
      if (!(a.size() == 2 && a.found(i0) && a.value(i0).is_integer() &&
            a.value(i0).to_integer() == 43 && a.found(i1) &&
            a.value(i1).is_hashmaparray())) {
        ans = false;
      } else {
        il::HashMapArray<il::String, il::Dynamic> &b =
            a.value(i1).as_hashmaparray();
        il::int_t j = b.search("c");
        if (!(b.size() == 1 && b.found(j) && b.value(j).is_hashmaparray())) {
          ans = false;
        } else {
          il::HashMapArray<il::String, il::Dynamic> &c =
              b.value(j).as_hashmaparray();
          il::int_t j0 = c.search("answer");
          if (!(c.size() == 1 && c.found(j0) && c.value(j0).is_integer() &&
                c.value(j0).to_integer() == 42)) {
            ans = false;
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, implicit_groups) {
  bool ans = true;

  il::String filename = directory;
  filename.append("implicit-groups.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a");
    if (!(config.found(i) && config.value(i).is_hashmaparray())) {
      ans = false;
    } else {
      il::HashMapArray<il::String, il::Dynamic> &a =
          config.value(i).as_hashmaparray();
      il::int_t i1 = a.search("b");
      if (!(a.size() == 1 && a.found(i1) && a.value(i1).is_hashmaparray())) {
        ans = false;
      } else {
        il::HashMapArray<il::String, il::Dynamic> &b =
            a.value(i1).as_hashmaparray();
        il::int_t j = b.search("c");
        if (!(b.size() == 1 && b.found(j) && b.value(j).is_hashmaparray())) {
          ans = false;
        } else {
          il::HashMapArray<il::String, il::Dynamic> &c =
              b.value(j).as_hashmaparray();
          il::int_t j0 = c.search("answer");
          if (!(c.size() == 1 && c.found(j0) && c.value(j0).is_integer() &&
                c.value(j0).to_integer() == 42)) {
            ans = false;
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, integer) {
  bool ans = true;

  il::String filename = directory;
  filename.append("integer.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("answer");
    il::int_t i1 = config.search("neganswer");
    if (!(config.found(i0) && config.value(i0).is_integer() &&
          config.value(i0).to_integer() == 42 && config.found(i1) &&
          config.value(i1).is_integer() &&
          config.value(i1).to_integer() == -42)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, key_equals_nospace) {
  bool ans = true;

  il::String filename = directory;
  filename.append("key-equals-nospace.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("answer");
    if (!(config.found(i) && config.value(i).is_integer() &&
          config.value(i).to_integer() == 42)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, key_space) {
  bool ans = true;

  il::String filename = directory;
  filename.append("key-space.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a b");
    if (!(config.found(i) && config.value(i).is_integer() &&
          config.value(i).to_integer() == 1)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, key_special_chars) {
  bool ans = true;

  il::String filename = directory;
  filename.append("key-special-chars.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("~!@$^&*()_+-`1234567890[]|/?><.,;:'");
    if (!(config.found(i) && config.value(i).is_integer() &&
          config.value(i).to_integer() == 1)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, long_floating_point) {
  bool ans = true;

  il::String filename = directory;
  filename.append("long-float.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("longpi");
    il::int_t i1 = config.search("neglongpi");
    if (!(config.found(i0) && config.value(i0).is_floating_point() &&
          config.value(i0).to_floating_point() == 3.141592653589793 &&
          config.found(i1) && config.value(i1).is_floating_point() &&
          config.value(i1).to_floating_point() == -3.141592653589793)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, long_integer) {
  bool ans = true;

  il::String filename = directory;
  filename.append("long-integer.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (status.is_error() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("answer");
    il::int_t i1 = config.search("neganswer");
    if (!(config.found(i0) && config.value(i0).is_integer() &&
          config.value(i0).to_integer() == 9223372036854775807 &&
          config.found(i1) && config.value(i1).is_integer() &&
          config.value(i1).to_integer() == (-9223372036854775807 - 1))) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}
