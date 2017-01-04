//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_DUMMY_TEST_H
#define IL_DUMMY_TEST_H

#include <il/Array.h>

class Dummy {
 public:
  static il::int_t current;
  static il::Array<il::int_t> destroyed;

 private:
  il::int_t id_;

 public:
  static void reset() {
    current = 0;
    destroyed.resize(0);
  }

 public:
  Dummy() {
    id_ = current;
    ++current;
  }
  Dummy(const Dummy& other) { id_ = other.id_; }
  Dummy(Dummy&& other) {
    id_ = other.id_;
    other.id_ = -(1 + other.id_);
  }
  Dummy& operator=(const Dummy& other) {
    id_ = other.id_;
    return *this;
  }
  Dummy& operator=(Dummy&& other) {
    id_ = other.id_;
    other.id_ = -(1001 + other.id_);
    return *this;
  }
  ~Dummy() { destroyed.append(id_); }
  il::int_t id() { return id_; };
};

#endif  // IL_DUMMY_TEST_H
