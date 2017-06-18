//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <gtest/gtest.h>

#include <il/container/info/Status.h>

//// You are not allowed to asked if it is ok until the status has been set
// TEST(Status, default_constructor) {
//  il::Status status{};
//
//  bool success = false;
//  try {
//    status.isOk();
//  } catch (...) {
//    success = true;
//  }
//
//  ASSERT_TRUE(success);
//}
//
// TEST(Status, check_error) {
//  il::Status status{};
//  status.setOk();
//
//  ASSERT_TRUE(status.isOk());
//}
//
//// You don't have to check is the status has not been set
// TEST(Status, destructor) {
//  {
//    il::Status  status{};
//  }
//  const bool success = true;
//
//  ASSERT_TRUE(success);
//}
//
//// We can't check if the status has to be verified because we cannot throw an
//// exception in the destructor
