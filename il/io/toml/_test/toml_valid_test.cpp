//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/toml.h>

il::String directory = IL_FOLDER "/io/toml/_test/valid/";

TEST(Toml, array_empty) {
  bool ans = true;

  il::String filename = directory;
  std::cout << directory.asCString() << std::endl;

  filename.append("array-empty.toml");

  il::Status status{};
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    const il::Location i0 = config.search("thevoid");
    if (!(config.found(i0) && config.value(i0).is<il::Array<il::Dynamic>>())) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array1 =
          config.value(i0).as<il::Array<il::Dynamic>>();
      if (array1.size() != 1 || !(array1[0].is<il::Array<il::Dynamic>>())) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &array2 =
            array1[0].as<il::Array<il::Dynamic>>();
        if (array2.size() != 1 || !(array2[0].is<il::Array<il::Dynamic>>())) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &array3 =
              array2[0].as<il::Array<il::Dynamic>>();
          if (array3.size() != 1 || !(array3[0].is<il::Array<il::Dynamic>>())) {
            ans = false;
          } else {
            const il::Array<il::Dynamic> &array4 =
                array3[0].as<il::Array<il::Dynamic>>();
            if (array4.size() != 1 ||
                !(array4[0].is<il::Array<il::Dynamic>>())) {
              ans = false;
            } else {
              const il::Array<il::Dynamic> &array5 =
                  array4[0].as<il::Array<il::Dynamic>>();
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    const il::Location i = config.search("ints");
    if (!(config.found(i) && config.value(i).is<il::Array<il::Dynamic>>())) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array =
          config.value(i).as<il::Array<il::Dynamic>>();
      if (array.size() != 3 || !array[0].is<il::int_t>() ||
          array[0].to<il::int_t>() != 1 || !array[1].is<il::int_t>() ||
          array[1].to<il::int_t>() != 2 || !array[2].is<il::int_t>() ||
          array[2].to<il::int_t>() != 3) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    const il::Location i = config.search("mixed");
    if (!(config.found(i) && config.value(i).is<il::Array<il::Dynamic>>() &&
          config.value(i).as<il::Array<il::Dynamic>>().size() == 3)) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array =
          config.value(i).as<il::Array<il::Dynamic>>();

      if (!(array[0].is<il::Array<il::Dynamic>>() &&
            array[0].as<il::Array<il::Dynamic>>().size() == 2)) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &subarray0 =
            array[0].as<il::Array<il::Dynamic>>();
        if (!(subarray0[0].is<il::int_t>() &&
              subarray0[0].to<il::int_t>() == 1 &&
              subarray0[1].is<il::int_t>() &&
              subarray0[1].to<il::int_t>() == 2)) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &subarray1 =
              array[1].as<il::Array<il::Dynamic>>();
          if (!(subarray1[0].is<il::String>() &&
                subarray1[0].as<il::String>() == "a" &&
                subarray1[1].is<il::String>() &&
                subarray1[1].as<il::String>() == "b")) {
            ans = false;
          } else {
            const il::Array<il::Dynamic> &subarray2 =
                array[2].as<il::Array<il::Dynamic>>();
            if (!(subarray2[0].is<double>() &&
                  subarray2[0].to<double>() == 1.1 &&
                  subarray2[1].is<double>() &&
                  subarray2[1].to<double>() == 2.1)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    const il::Location i = config.search("nest");
    if (!(config.found(i) && config.value(i).is<il::Array<il::Dynamic>>() &&
          config.value(i).as<il::Array<il::Dynamic>>().size() == 2)) {
      ans = false;
    } else {
      const il::Array<il::Dynamic> &array =
          config.value(i).as<il::Array<il::Dynamic>>();

      if (!(array[0].is<il::Array<il::Dynamic>>() &&
            array[0].as<il::Array<il::Dynamic>>().size() == 1 &&
            array[1].is<il::Array<il::Dynamic>>() &&
            array[1].as<il::Array<il::Dynamic>>().size() == 1)) {
        ans = false;
      } else {
        const il::Array<il::Dynamic> &subarray0 =
            array[0].as<il::Array<il::Dynamic>>();
        if (!(subarray0[0].is<il::String>() &&
              subarray0[0].as<il::String>() == "a")) {
          ans = false;
        } else {
          const il::Array<il::Dynamic> &subarray1 =
              array[1].as<il::Array<il::Dynamic>>();
          if (!(subarray1[0].is<il::String>() &&
                subarray1[0].as<il::String>() == "b")) {
            ans = false;
          }
        }
      }
    }
  }

  ASSERT_TRUE(ans);
}

TEST(Toml, bool) {
  bool ans = true;

  il::String filename = directory;
  filename.append("bool.toml");

  il::Status status{};
  auto config =
      il::load<il::MapArray<il::String, il::Dynamic>>(filename, il::io, status);
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    const il::Location i = config.search("t");
    if (!(config.found(i) && config.value(i).is<bool>() &&
          config.value(i).as<bool>())) {
      ans = false;
    } else {
      const il::Location i = config.search("f");
      if (!(config.found(i) && config.value(i).is<bool>() &&
            config.value(i).as<bool>() == false)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    const il::Location i = config.search("group");
    if (!(config.found(i) &&
          config.value(i).is<il::MapArray<il::String, il::Dynamic>>())) {
      ans = false;
    } else {
      il::MapArray<il::String, il::Dynamic> &group =
          config.value(i).as<il::MapArray<il::String, il::Dynamic>>();
      il::Location j0 = group.search("answer");
      il::Location j1 = group.search("more");
      if (!(group.size() == 2 && group.found(j0) &&
            group.value(j0).is<il::int_t>() &&
            group.value(j0).to<il::int_t>() == 42 && group.found(j1) &&
            group.value(j1).is<il::Array<il::Dynamic>>())) {
        ans = false;
      } else {
        il::Array<il::Dynamic> &array =
            group.value(j1).as<il::Array<il::Dynamic>>();
        if (!(array.size() == 2 && array[0].is<il::int_t>() &&
              array[0].to<il::int_t>() == 42 && array[1].is<il::int_t>() &&
              array[1].to<il::int_t>() == 42)) {
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
  if (!status.ok() || config.size() != 0) {
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
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    il::Location i0 = config.search("pi");
    il::Location i1 = config.search("negpi");
    if (!(config.found(i0) && config.value(i0).is<double>() &&
          config.value(i0).to<double>() == 3.14 && config.found(i1) &&
          config.value(i1).is<double>() &&
          config.value(i1).to<double>() == -3.14)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    il::Location i = config.search("a");
    if (!(config.found(i) &&
          config.value(i).is<il::MapArray<il::String, il::Dynamic>>())) {
      ans = false;
    } else {
      const il::MapArray<il::String, il::Dynamic> &a =
          config.value(i).as<il::MapArray<il::String, il::Dynamic>>();
      il::Location i0 = a.search("better");
      il::Location i1 = a.search("b");
      if (!(a.size() == 2 && a.found(i0) && a.value(i0).is<il::int_t>() &&
            a.value(i0).to<il::int_t>() == 43 && a.found(i1) &&
            a.value(i1).is<il::MapArray<il::String, il::Dynamic>>())) {
        ans = false;
      } else {
        const il::MapArray<il::String, il::Dynamic> &b =
            a.value(i1).as<il::MapArray<il::String, il::Dynamic>>();
        il::Location j = b.search("c");
        if (!(b.size() == 1 && b.found(j) &&
              b.value(j).is<il::MapArray<il::String, il::Dynamic>>())) {
          ans = false;
        } else {
          const il::MapArray<il::String, il::Dynamic> &c =
              b.value(j).as<il::MapArray<il::String, il::Dynamic>>();
          il::Location j0 = c.search("answer");
          if (!(c.size() == 1 && c.found(j0) && c.value(j0).is<il::int_t>() &&
                c.value(j0).to<il::int_t>() == 42)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    il::Location i = config.search("a");
    if (!(config.found(i) &&
          config.value(i).is<il::MapArray<il::String, il::Dynamic>>())) {
      ans = false;
    } else {
      const il::MapArray<il::String, il::Dynamic> &a =
          config.value(i).as<il::MapArray<il::String, il::Dynamic>>();
      il::Location i0 = a.search("better");
      il::Location i1 = a.search("b");
      if (!(a.size() == 2 && a.found(i0) && a.value(i0).is<il::int_t>() &&
            a.value(i0).to<il::int_t>() == 43 && a.found(i1) &&
            a.value(i1).is<il::MapArray<il::String, il::Dynamic>>())) {
        ans = false;
      } else {
        const il::MapArray<il::String, il::Dynamic> &b =
            a.value(i1).as<il::MapArray<il::String, il::Dynamic>>();
        il::Location j = b.search("c");
        if (!(b.size() == 1 && b.found(j) &&
              b.value(j).is<il::MapArray<il::String, il::Dynamic>>())) {
          ans = false;
        } else {
          const il::MapArray<il::String, il::Dynamic> &c =
              b.value(j).as<il::MapArray<il::String, il::Dynamic>>();
          il::Location j0 = c.search("answer");
          if (!(c.size() == 1 && c.found(j0) && c.value(j0).is<il::int_t>() &&
                c.value(j0).to<il::int_t>() == 42)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    il::Location i = config.search("a");
    if (!(config.found(i) &&
          config.value(i).is<il::MapArray<il::String, il::Dynamic>>())) {
      ans = false;
    } else {
      const il::MapArray<il::String, il::Dynamic> &a =
          config.value(i).as<il::MapArray<il::String, il::Dynamic>>();
      il::Location i1 = a.search("b");
      if (!(a.size() == 1 && a.found(i1) &&
            a.value(i1).is<il::MapArray<il::String, il::Dynamic>>())) {
        ans = false;
      } else {
        const il::MapArray<il::String, il::Dynamic> &b =
            a.value(i1).as<il::MapArray<il::String, il::Dynamic>>();
        il::Location j = b.search("c");
        if (!(b.size() == 1 && b.found(j) &&
              b.value(j).is<il::MapArray<il::String, il::Dynamic>>())) {
          ans = false;
        } else {
          const il::MapArray<il::String, il::Dynamic> &c =
              b.value(j).as<il::MapArray<il::String, il::Dynamic>>();
          il::Location j0 = c.search("answer");
          if (!(c.size() == 1 && c.found(j0) && c.value(j0).is<il::int_t>() &&
                c.value(j0).to<il::int_t>() == 42)) {
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
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    il::Location i0 = config.search("answer");
    il::Location i1 = config.search("neganswer");
    if (!(config.found(i0) && config.value(i0).is<il::int_t>() &&
          config.value(i0).to<il::int_t>() == 42 && config.found(i1) &&
          config.value(i1).is<il::int_t>() &&
          config.value(i1).to<il::int_t>() == -42)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    il::Location i = config.search("answer");
    if (!(config.found(i) && config.value(i).is<il::int_t>() &&
          config.value(i).to<il::int_t>() == 42)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    il::Location i = config.search("a b");
    if (!(config.found(i) && config.value(i).is<il::int_t>() &&
          config.value(i).to<il::int_t>() == 1)) {
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
  if (!status.ok() || config.size() != 1) {
    ans = false;
  } else {
    il::Location i = config.search("~!@$^&*()_+-`1234567890[]|/?><.,;:'");
    if (!(config.found(i) && config.value(i).is<il::int_t>() &&
          config.value(i).to<il::int_t>() == 1)) {
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
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    il::Location i0 = config.search("longpi");
    il::Location i1 = config.search("neglongpi");
    if (!(config.found(i0) && config.value(i0).is<double>() &&
          config.value(i0).to<double>() == 3.141592653589793 &&
          config.found(i1) && config.value(i1).is<double>() &&
          config.value(i1).to<double>() == -3.141592653589793)) {
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
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    il::Location i0 = config.search("answer");
    il::Location i1 = config.search("neganswer");
    if (!(config.found(i0) && config.value(i0).is<il::int_t>() &&
          config.value(i0).to<il::int_t>() == 9223372036854775807 &&
          config.found(i1) && config.value(i1).is<il::int_t>() &&
          config.value(i1).to<il::int_t>() == (-9223372036854775807 - 1))) {
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
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    il::Location i0 = config.search("input_directory");
    il::Location i1 = config.search("Young_modulus");
    if (!(config.found(i0) && config.value(i0).is<il::String>() &&
          config.value(i0).as<il::String>() == "Mesh_Files" &&
          config.found(i1) && config.value(i1).is<double>() &&
          config.value(i1).to<double>() == 1.0)) {
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
  if (!status.ok() || config.size() != 2) {
    ans = false;
  } else {
    il::Location i0 = config.search("a");
    il::Location i1 = config.search("b");
    if (!(config.found(i0) && config.value(i0).is<double>() &&
          config.value(i0).to<double>() == 0.0 && config.found(i1) &&
          config.value(i1).is<il::int_t>() &&
          config.value(i1).to<il::int_t>() == 0)) {
      ans = false;
    }
  }

  ASSERT_TRUE(ans);
}
