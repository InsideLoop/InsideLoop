//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/Array.h>
#include <il/container/1d/_test/Dummy_test.h>

TEST(Array, default_constructor) {
  il::Array<il::int_t> v{};

  ASSERT_TRUE(v.size() == 0 && v.capacity() == 0 && v.data() == nullptr);
}

TEST(Array, size_constructor_0) {
  il::Array<il::int_t> v{0};

  ASSERT_TRUE(v.size() == 0 && v.capacity() == 0 && v.data() == nullptr);
}

TEST(Array, size_constructor_1) {
  const il::int_t n{3};
  il::Array<il::int_t> v{n};

  ASSERT_TRUE(v.size() == n && v.capacity() == n);
}

TEST(Array, size_value_constructor_0) {
  const il::int_t x{5};
  il::Array<il::int_t> v{0, x};

  ASSERT_TRUE(v.size() == 0 && v.capacity() == 0 && v.data() == nullptr);
}

TEST(Array, size_value_constructor_1) {
  const il::int_t n{3};
  const il::int_t x{5};
  il::Array<il::int_t> v{n, x};

  bool correct_elements{true};
  for (il::int_t i{0}; i < n; ++i) {
    if (v[i] != x) {
      correct_elements = false;
    }
  }

  ASSERT_TRUE(v.size() == n && v.capacity() == n && correct_elements);
}

TEST(Array, initializer_list_constructor_0) {
  il::Array<il::int_t> v{il::value, {}};

  ASSERT_TRUE(v.size() == 0 && v.capacity() == 0 && v.data() == nullptr);
}

TEST(Array, initializer_list_constructor_1) {
  il::Array<il::int_t> v{il::value, {7, 8, 9}};

  ASSERT_TRUE(v.size() == 3 && v.capacity() == 3 && v[0] == 7 && v[1] == 8 &&
              v[2] == 9);
}

TEST(Array, copy_constructor) {
  const il::int_t n{3};
  il::Array<il::int_t> v{n};
  for (il::int_t i{0}; i < n; ++i) {
    v[i] = i;
  }
  v.reserve(2 * n);

  il::Array<il::int_t> w{v};
  bool same_elements{true};
  for (il::int_t i{0}; i < n; ++i) {
    if (w[i] != i) {
      same_elements = false;
    }
  }

  ASSERT_TRUE(w.size() == n && w.capacity() == n && same_elements);
}

TEST(Array, move_constructor) {
  const il::int_t n{3};
  il::Array<il::int_t> v{n};
  for (il::int_t i{0}; i < n; ++i) {
    v[i] = i;
  }
  v.reserve(2 * n);
  il::int_t* old_v_data{v.data()};

  il::Array<il::int_t> w{std::move(v)};
  bool same_elements{true};
  for (il::int_t i{0}; i < n; ++i) {
    if (w[i] != i) {
      same_elements = false;
    }
  }

  ASSERT_TRUE(v.data() == nullptr && v.size() == 0 && v.capacity() == 0 &&
              w.data() == old_v_data && w.size() == n &&
              w.capacity() == 2 * n && same_elements);
}

TEST(Array, copy_assignment_0) {
  const il::int_t n{3};
  il::Array<il::int_t> v{n};
  for (il::int_t i{0}; i < n; ++i) {
    v[i] = i;
  }
  v.reserve(2 * n);

  il::Array<il::int_t> w{};
  w = v;
  bool same_elements{true};
  for (il::int_t i{0}; i < n; ++i) {
    if (w[i] != i) {
      same_elements = false;
    }
  }

  ASSERT_TRUE(w.size() == n && w.capacity() == n && same_elements);
}

TEST(Array, copy_assignment_1) {
  const il::int_t n{3};
  il::Array<il::int_t> v{n};
  for (il::int_t i{0}; i < n; ++i) {
    v[i] = i;
  }
  v.reserve(2 * n);

  il::Array<il::int_t> w{3 * n};
  w = v;
  bool same_elements{true};
  for (il::int_t i{0}; i < n; ++i) {
    if (w[i] != i) {
      same_elements = false;
    }
  }

  ASSERT_TRUE(w.size() == n && w.capacity() == 3 * n && same_elements);
}

TEST(Array, copy_assignment_object_0) {
  Dummy::reset();

  il::Array<Dummy> v{5};
  il::Array<Dummy> w{3};
  v = w;

  ASSERT_TRUE(Dummy::destroyed.size() == 2 && Dummy::destroyed[0] == 4 &&
              Dummy::destroyed[1] == 3);
}

TEST(Array, copy_assignment_object_1) {
  Dummy::reset();

  il::Array<Dummy> v{3};
  il::Array<Dummy> w{5};
  v = w;

  ASSERT_TRUE(Dummy::destroyed.size() == 3 && Dummy::destroyed[0] == 2 &&
              Dummy::destroyed[1] == 1 && Dummy::destroyed[2] == 0);
}

TEST(Array, move_assignment) {
  const il::int_t n{3};
  il::Array<il::int_t> v{n};
  for (il::int_t i{0}; i < n; ++i) {
    v[i] = i;
  }
  v.reserve(2 * n);
  il::int_t* old_v_data{v.data()};

  il::Array<il::int_t> w{3 * n};
  w = std::move(v);
  bool same_elements{true};
  for (il::int_t i{0}; i < n; ++i) {
    if (w[i] != i) {
      same_elements = false;
    }
  }

  ASSERT_TRUE(v.data() == nullptr && v.size() == 0 && v.capacity() == 0 &&
              w.data() == old_v_data && w.size() == n &&
              w.capacity() == 2 * n && same_elements);
}

TEST(Array, move_assignment_object_0) {
  Dummy::reset();

  il::Array<Dummy> v{3};
  il::Array<Dummy> w{5};
  v = std::move(w);

  ASSERT_TRUE(Dummy::destroyed.size() == 3 && Dummy::destroyed[0] == 2 &&
              Dummy::destroyed[1] == 1 && Dummy::destroyed[2] == 0);
}

TEST(Array, destructor_object) {
  Dummy::reset();
  {
    const il::int_t n{3};
    il::Array<Dummy> v{n};
  }

  ASSERT_TRUE(Dummy::destroyed.size() == 3 && Dummy::destroyed[0] == 2 &&
              Dummy::destroyed[1] == 1 && Dummy::destroyed[2] == 0);
}

TEST(Array, bounds_checking) {
  bool test_passed{true};

  for (il::int_t n : {0, 1}) {
    il::Array<double> v{n};
    bool local_test_passed;

    try {
      local_test_passed = false;
      (void)v[-1];
    } catch (il::abort_exception) {
      local_test_passed = true;
    }
    if (!local_test_passed) {
      test_passed = false;
    }

    try {
      local_test_passed = false;
      (void)v[n];
    } catch (il::abort_exception) {
      local_test_passed = true;
    }
    if (!local_test_passed) {
      test_passed = false;
    }
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_default) {
  bool test_passed{true};

  il::Array<double> v{};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)v[-1];
  } catch (il::abort_exception) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)v[0];
  } catch (il::abort_exception) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_const) {
  bool test_passed{true};

  for (il::int_t n : {0, 1}) {
    const il::Array<double> v{n};
    bool local_test_passed;

    try {
      local_test_passed = false;
      (void)v[-1];
    } catch (il::abort_exception) {
      local_test_passed = true;
    }
    if (!local_test_passed) {
      test_passed = false;
    }

    try {
      local_test_passed = false;
      (void)v[n];
    } catch (il::abort_exception) {
      local_test_passed = true;
    }
    if (!local_test_passed) {
      test_passed = false;
    }
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_default_const) {
  bool test_passed{true};

  const il::Array<double> v{};
  bool local_test_passed;

  try {
    local_test_passed = false;
    (void)v[-1];
  } catch (il::abort_exception) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  try {
    local_test_passed = false;
    (void)v[0];
  } catch (il::abort_exception) {
    local_test_passed = true;
  }
  if (!local_test_passed) {
    test_passed = false;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_back) {
  il::Array<double> v{0};
  bool test_passed{false};

  try {
    (void)v.back();
  } catch (il::abort_exception) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_back_default) {
  il::Array<double> v{};
  bool test_passed{false};

  try {
    (void)v.back();
  } catch (il::abort_exception) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_back_const) {
  const il::Array<double> v{0};
  bool test_passed{false};

  try {
    (void)v.back();
  } catch (il::abort_exception) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, bounds_checking_back_default_const) {
  const il::Array<double> v{};
  bool test_passed{false};

  try {
    (void)v.back();
  } catch (il::abort_exception) {
    test_passed = true;
  }

  ASSERT_TRUE(test_passed);
}

TEST(Array, resize_0) {
  il::Array<il::int_t> v{il::value, {6, 7, 8, 9}};
  il::int_t* const old_v_data{v.data()};
  v.resize(3);

  ASSERT_TRUE(old_v_data == v.data() && v.size() == 3 && v.capacity() == 4 &&
              v[0] == 6 && v[1] == 7 && v[2] == 8);
}

TEST(Array, resize_1) {
  il::Array<il::int_t> v{il::value, {6, 7, 8}};
  v.resize(4);

  ASSERT_TRUE(v.size() == 4 && v[0] == 6 && v[1] == 7 && v[2] == 8);
}

TEST(Array, resize_object_0) {
  Dummy::reset();
  il::Array<Dummy> v{5};
  v.resize(3);

  ASSERT_TRUE(Dummy::destroyed.size() == 2 && Dummy::destroyed[0] == 4 &&
              Dummy::destroyed[1] == 3);
}

TEST(Array, resize_object_1) {
  Dummy::reset();
  il::Array<Dummy> v{3};
  v.resize(5);

  ASSERT_TRUE(Dummy::destroyed.size() == 3 && Dummy::destroyed[0] == -(1 + 2) &&
              Dummy::destroyed[1] == -(1 + 1) &&
              Dummy::destroyed[2] == -(1 + 0));
}

TEST(Array, reserve) {
  il::Array<il::int_t> v{};
  v.reserve(3);
  il::int_t* const old_v_data{v.data()};
  v.append(7);
  v.append(8);
  v.append(9);
  il::int_t* const new_v_data{v.data()};

  ASSERT_TRUE(old_v_data == new_v_data);
}

TEST(Array, reserve_object) {
  Dummy::reset();
  il::Array<Dummy> v{3};
  v.reserve(5);

  ASSERT_TRUE(Dummy::destroyed.size() == 3 && Dummy::destroyed[0] == -(1 + 2) &&
              Dummy::destroyed[1] == -(1 + 1) &&
              Dummy::destroyed[2] == -(1 + 0));
}

TEST(Array, append_0) {
  il::Array<il::int_t> v{};
  v.append(5);
  v.append(6);
  v.append(7);
  v.append(8);
  v.append(9);

  ASSERT_TRUE(v.size() == 5 && v[0] == 5 && v[1] == 6 && v[2] == 7 &&
              v[3] == 8 && v[4] == 9);
}

TEST(Array, append_1) {
  il::Array<il::int_t> v{};
  v.reserve(4);
  v.resize(3);
  il::int_t* const old_v_data{v.data()};
  v.append(0);
  il::int_t* const new_v_data{v.data()};

  ASSERT_TRUE(old_v_data == new_v_data);
}

TEST(Array, append_object) {
  Dummy::reset();
  il::Array<Dummy> v{3};
  v.append(Dummy{});

  ASSERT_TRUE(Dummy::destroyed.size() == 4 && Dummy::destroyed[0] == -(1 + 2) &&
              Dummy::destroyed[1] == -(1 + 1) &&
              Dummy::destroyed[2] == -(1 + 0) &&
              Dummy::destroyed[3] == -(1 + 3));
}

TEST(Array, emplace_back) {
  Dummy::reset();
  il::Array<Dummy> v{3};
  v.append(il::emplace, il::arg());

  ASSERT_TRUE(Dummy::destroyed.size() == 3 && Dummy::destroyed[0] == -(1 + 2) &&
              Dummy::destroyed[1] == -(1 + 1) &&
              Dummy::destroyed[2] == -(1 + 0));
}
