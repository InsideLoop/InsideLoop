//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/norm.h>

#ifdef IL_BLAS
TEST(norm_staticarray, L1) {
  il::StaticArray<double, 3> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L1), 3.5);
}

TEST(norm_staticarray, L2) {
  il::StaticArray<double, 3> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L2), std::sqrt(5.25));
}

TEST(norm_staticarray, Linf) {
  il::StaticArray<double, 3> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::Linf), 2.0);
}

TEST(norm_array, L1) {
  il::Array<double> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L1), 3.5);
}

TEST(norm_array, L2) {
  il::Array<double> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::L2), std::sqrt(5.25));
}

TEST(norm_array, Linf) {
  il::Array<double> v{il::value, {-2.0, 0.5, 1.0}};

  ASSERT_DOUBLE_EQ(il::norm(v, il::Norm::Linf), 2.0);
}
#endif
