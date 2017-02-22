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

il::String directory =
    "/home/fayard/Documents/Projects/InsideLoop/InsideLoop/il/io/toml/_test/"
    "valid/";

TEST(Toml, array_empty) {
  bool ans = true;

  il::String filename = directory;
  filename.append("array-empty.toml");

  il::Status status{};
  il::Toml config = il::load<il::Toml>(filename, il::io, status);
  if (!status.ok() || config.size() != 1) {
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
  if (!status.ok() || config.size() != 1) {
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
