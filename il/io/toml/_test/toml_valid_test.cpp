//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/toml.h>

#include <iostream>

il::String directory =
    "/Users/fayard/Documents/Projects/InsideLoop/InsideLoop/il/io/toml/_test/"
    "valid/";

TEST(Toml, array_empty) {
  bool ans = true;

  il::String filename = directory;
  filename.append("array-empty.toml");

  il::Status status{};
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i0 = config.search("thevoid");
    if (!(config.hasFound(i0) && config.value(i0).isArray())) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array1 = config.value(i0).asArray();
      if (array1.size() != 1 || !(array1[0].isArray())) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &array2 = array1[0].asArray();
        if (array2.size() != 1 || !(array2[0].isArray())) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &array3 = array2[0].asArray();
          if (array3.size() != 1 || !(array3[0].isArray())) {
            ans = false;
          } else {
            const il::Array<il::Dynamic> &array4 = array3[0].asArray();
            if (array4.size() != 1 || !(array4[0].isArray())) {
              ans = false;
            } else {
              const il::Array<il::Dynamic> &array5 = array4[0].asArray();
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("ints");
    if (!(config.hasFound(i) && config.value(i).isArray())) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array = config.value(i).asArray();
      if (array.size() != 3 || !array[0].isInteger() ||
          array[0].toInteger() != 1 || !array[1].isInteger() ||
          array[1].toInteger() != 2 || !array[2].isInteger() ||
          array[2].toInteger() != 3) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("mixed");
    if (!(config.hasFound(i) && config.value(i).isArray() &&
          config.value(i).asArray().size() == 3)) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array = config.value(i).asArray();

      if (!(array[0].isArray() && array[0].asArray().size() == 2)) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &subarray0 = array[0].asArray();
        if (!(subarray0[0].isInteger() && subarray0[0].toInteger() == 1 &&
              subarray0[1].isInteger() && subarray0[1].toInteger() == 2)) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &subarray1 = array[1].asArray();
          if (!(subarray1[0].isString() && subarray1[0].asString() == "a" &&
                subarray1[1].isString() && subarray1[1].asString() == "b")) {
            ans = false;
          } else {
            const il::Array<il::Dynamic> &subarray2 = array[2].asArray();
            if (!(subarray2[0].isDouble() && subarray2[0].toDouble() == 1.1 &&
                  subarray2[1].isDouble() && subarray2[1].toDouble() == 2.1)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("nest");
    if (!(config.hasFound(i) && config.value(i).isArray() &&
          config.value(i).asArray().size() == 2)) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array = config.value(i).asArray();

      if (!(array[0].isArray() && array[0].asArray().size() == 1 &&
            array[1].isArray() && array[1].asArray().size() == 1)) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &subarray0 = array[0].asArray();
        if (!(subarray0[0].isString() && subarray0[0].asString() == "a")) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &subarray1 = array[1].asArray();
          if (!(subarray1[0].isString() && subarray1[0].asString() == "b")) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    const il::int_t i = config.search("t");
    if (!(config.hasFound(i) && config.value(i).isBool() &&
          config.value(i).toBool())) {
      ans = false;
    } else {
      const il::int_t i = config.search("f");
      if (!(config.hasFound(i) && config.value(i).isBool() &&
            config.value(i).toBool() == false)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    const il::int_t i = config.search("group");
    if (!(config.hasFound(i) && config.value(i).isMapArray())) {
      ans = false;
    } else {
      il::MapArray<il::String, il::Dynamic> &group =
          config.value(i).asMapArray();
      il::int_t j0 = group.search("answer");
      il::int_t j1 = group.search("more");
      if (!(group.size() == 2 && group.hasFound(j0) &&
            group.value(j0).isInteger() && group.value(j0).toInteger() == 42 &&
            group.hasFound(j1) && group.value(j1).isArray())) {
        ans = false;
      } else {
        il::Array<il::Dynamic> &array = group.value(j1).asArray();
        if (!(array.size() == 2 && array[0].isInteger() &&
              array[0].toInteger() == 42 && array[1].isInteger() &&
              array[1].toInteger() == 42)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 0) {
    ans = false;
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, double) {
  bool ans = true;

  il::String filename = directory;
  filename.append("float.toml");

  il::Status status{};
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("pi");
    il::int_t i1 = config.search("negpi");
    if (!(config.hasFound(i0) && config.value(i0).isDouble() &&
          config.value(i0).toDouble() == 3.14 && config.hasFound(i1) &&
          config.value(i1).isDouble() &&
          config.value(i1).toDouble() == -3.14)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a");
    if (!(config.hasFound(i) && config.value(i).isMapArray())) {
      ans = false;
    } else {
      const il::MapArray<il::String, il::Dynamic> &a =
          config.value(i).asMapArray();
      il::int_t i0 = a.search("better");
      il::int_t i1 = a.search("b");
      if (!(a.size() == 2 && a.hasFound(i0) && a.value(i0).isInteger() &&
            a.value(i0).toInteger() == 43 && a.hasFound(i1) &&
            a.value(i1).isMapArray())) {
        ans = false;
      } else {
        const il::MapArray<il::String, il::Dynamic> &b =
            a.value(i1).asMapArray();
        il::int_t j = b.search("c");
        if (!(b.size() == 1 && b.hasFound(j) && b.value(j).isMapArray())) {
          ans = false;
        } else {
          const il::MapArray<il::String, il::Dynamic> &c =
              b.value(j).asMapArray();
          il::int_t j0 = c.search("answer");
          if (!(c.size() == 1 && c.hasFound(j0) && c.value(j0).isInteger() &&
                c.value(j0).toInteger() == 42)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a");
    if (!(config.hasFound(i) && config.value(i).isMapArray())) {
      ans = false;
    } else {
      const il::MapArray<il::String, il::Dynamic> &a =
          config.value(i).asMapArray();
      il::int_t i0 = a.search("better");
      il::int_t i1 = a.search("b");
      if (!(a.size() == 2 && a.hasFound(i0) && a.value(i0).isInteger() &&
            a.value(i0).toInteger() == 43 && a.hasFound(i1) &&
            a.value(i1).isMapArray())) {
        ans = false;
      } else {
        const il::MapArray<il::String, il::Dynamic> &b =
            a.value(i1).asMapArray();
        il::int_t j = b.search("c");
        if (!(b.size() == 1 && b.hasFound(j) && b.value(j).isMapArray())) {
          ans = false;
        } else {
          const il::MapArray<il::String, il::Dynamic> &c =
              b.value(j).asMapArray();
          il::int_t j0 = c.search("answer");
          if (!(c.size() == 1 && c.hasFound(j0) && c.value(j0).isInteger() &&
                c.value(j0).toInteger() == 42)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a");
    if (!(config.hasFound(i) && config.value(i).isMapArray())) {
      ans = false;
    } else {
      const il::MapArray<il::String, il::Dynamic> &a =
          config.value(i).asMapArray();
      il::int_t i1 = a.search("b");
      if (!(a.size() == 1 && a.hasFound(i1) && a.value(i1).isMapArray())) {
        ans = false;
      } else {
        const il::MapArray<il::String, il::Dynamic> &b =
            a.value(i1).asMapArray();
        il::int_t j = b.search("c");
        if (!(b.size() == 1 && b.hasFound(j) && b.value(j).isMapArray())) {
          ans = false;
        } else {
          const il::MapArray<il::String, il::Dynamic> &c =
              b.value(j).asMapArray();
          il::int_t j0 = c.search("answer");
          if (!(c.size() == 1 && c.hasFound(j0) && c.value(j0).isInteger() &&
                c.value(j0).toInteger() == 42)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("answer");
    il::int_t i1 = config.search("neganswer");
    if (!(config.hasFound(i0) && config.value(i0).isInteger() &&
          config.value(i0).toInteger() == 42 && config.hasFound(i1) &&
          config.value(i1).isInteger() &&
          config.value(i1).toInteger() == -42)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("answer");
    if (!(config.hasFound(i) && config.value(i).isInteger() &&
          config.value(i).toInteger() == 42)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("a b");
    if (!(config.hasFound(i) && config.value(i).isInteger() &&
          config.value(i).toInteger() == 1)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 1) {
    ans = false;
  } else {
    il::int_t i = config.search("~!@$^&*()_+-`1234567890[]|/?><.,;:'");
    if (!(config.hasFound(i) && config.value(i).isInteger() &&
          config.value(i).toInteger() == 1)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("longpi");
    il::int_t i1 = config.search("neglongpi");
    if (!(config.hasFound(i0) && config.value(i0).isDouble() &&
          config.value(i0).toDouble() == 3.141592653589793 &&
          config.hasFound(i1) && config.value(i1).isDouble() &&
          config.value(i1).toDouble() == -3.141592653589793)) {
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
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("answer");
    il::int_t i1 = config.search("neganswer");
    if (!(config.hasFound(i0) && config.value(i0).isInteger() &&
          config.value(i0).toInteger() == 9223372036854775807 &&
          config.hasFound(i1) && config.value(i1).isInteger() &&
          config.value(i1).toInteger() == (-9223372036854775807 - 1))) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, windows_lines) {
  bool ans = true;

  il::String filename = directory;
  filename.append("windows-lines.toml");

  il::Status status{};
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("input_directory");
    il::int_t i1 = config.search("Young_modulus");
    if (!(config.hasFound(i0) && config.value(i0).isString() &&
          config.value(i0).asString() == "Mesh_Files" && config.hasFound(i1) &&
          config.value(i1).isDouble() && config.value(i1).toDouble() == 1.0)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, zero) {
  bool ans = true;

  il::String filename = directory;
  filename.append("zero.toml");

  il::Status status{};
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (status.notOk() || config.size() != 2) {
    ans = false;
  } else {
    il::int_t i0 = config.search("a");
    il::int_t i1 = config.search("b");
    if (!(config.hasFound(i0) && config.value(i0).isDouble() &&
          config.value(i0).toDouble() == 0.0 && config.hasFound(i1) &&
          config.value(i1).isInteger() && config.value(i1).toInteger() == 0)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}
