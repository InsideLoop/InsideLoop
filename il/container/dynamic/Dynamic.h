//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_DYNAMIC_H
#define IL_DYNAMIC_H

#include <il/Array.h>
#include <il/HashMapArray.h>
#include <il/String.h>
#include <il/container/dynamic/Type.h>

namespace il {

enum class DynamicType {
  null,
  boolean,
  integer,
  floating_point,
  string,
  array,
  hashmaparray
};

// A NaN is a number for which all the exponent bits are set to 1 and the
// payload is not equal to 0 (when the payload is equal to 0, we have a number
// infinity).
//
// === Generated NaN ===
//
// 000...000||0001[1111|1111|111]1: NaN coming from 0.0 /0.0 Gcc (4.8.5 & 6.3.0)
// 000...000||0001[1111|1111|111]0: NaN coming from 0.0 /0.0 Intel (17.0.1)
// 000...000||0001[1111|1111|111]0: NaN coming from 0.0 /0.0 Clang (3.9.1)
// 000...000||0001[1111|1111|111]1: NaN coming from 0.0 /0.0 MSVC
// 000...000||0001[1111|1111|111]0: NaN coming from 0.0 /0.0 ARM, Apple
//                                  Clang (8.0.0)
//
// === Quiet NaN ===
//
// 000...000||0001[1111|1111|111]0: quiet NAN (from limits) on Gcc (4.8.5 &
//                                  6.3.0), Intel (17.0.1), Clang (3.9.1) and
//                                  MSVC
//
// === Signaling NaN ===
//
// 000...000||0010[1111|1111|111]0: signaling NAN (from limits) on Gcc (4.8.5 &
//                                  6.3.0), Intel (17.0.1), Clang (3.9.1)
// 100...000||0001[1111|1111|111]0: signaling NAN (from limits) on MSVC
//                                  It does not "respect" the IEEE 754-2008
//                                  standard that suggests that if the highest
//                                  bit of the 52 bits is a 1, it is a quiet NaN
//
// === Signaling NaN after going through a function ===
//
// 000...000||0011[1111|1111|111]0: signaling NAN becoming a quiet NaN after
//                                  std::sin on Gcc (4.8.5 & 6.3.0), Intel
//                                  (17.0.1) and Clang (3.9.1)
// 100...000||0001[1111|1111|111]0: signaling NAN becoming a after std::sin
//
// === Infinity ===
// Infinities are not NaN but they have all their exponent bits set to 1
//
// 000...000||0000[1111|1111|111]0: +infinity
// 000...000||0000[1111|1111|111]1: -infinity

// =======================================================
// 000...000||1xxx[1111|1111|111]x: a type with a pointer.
//                                  It gives us 16 types.
// =======================================================
//
// 000...000||1000[1111|1111|111]0: a string
// 000...000||1100[1111|1111|111]0: a 1D array of il::dynamic
// 000...000||1010[1111|1111|111]0: <**> a 1D array of int
// 000...000||1110[1111|1111|111]0: <**> a 1D array of il::int_t
// 000...000||1001[1111|1111|111]0: <**> a 1D array of float
// 000...000||1101[1111|1111|111]0: <**> a 1D array of double
// 000...000||1011[1111|1111|111]0: <**> a 2D array of int
// 000...000||1111[1111|1111|111]0: <**> a 2D array of il::int_t
// 000...000||1000[1111|1111|111]1: <**> a 2D array of float
// 000...000||1100[1111|1111|111]1: <**> a 2D array of double
// 000...000||1010[1111|1111|111]1: a HashMapArray of <il::String, il::Dynamic>
// 000...000||1110[1111|1111|111]1: <**> a HashMapArray of <il::String,
// il::Dynamic> 000...000||1001[1111|1111|111]1: <**> a HashSet of int
// 000...000||1101[1111|1111|111]1: <**> a HashSet of il::int_t
// 000...000||1011[1111|1111|111]1: <**> a HashSet of float
// 000...000||1111[1111|1111|111]1: a 64-bit integer
//
// ==========================================================
// 000...000||01xx[1111|1111|111]x: a type without a pointer.
//                                  It gives us 8 types.
// ==========================================================
//
// 000...000||0100[1111|1111|111]0: null
// 000...000||0110[1111|1111|111]0: boolean
// 000...000||0101[1111|1111|111]0: a 48-bit integer
// 000...000||0111[1111|1111|111]0: a 48-bit positive integer
// 000...000||0100[1111|1111|111]1: free slot
// 000...000||0110[1111|1111|111]1: free slot
// 000...000||0101[1111|1111|111]1: empty on hash table
// 000...000||0111[1111|1111|111]1: tombstone on hash table

class Dynamic {
 private:
  union {
    // For bit access to get the type of the object and to copy trivial
    // types
    std::uint16_t data2_[4];
    std::uint64_t data8_;

    // For embedded types in the payload
    bool boolean_;
    std::int64_t n_;
    double x_;
    void* p_;
  };

 public:
  Dynamic();
  Dynamic(bool value);
  Dynamic(int n);
  Dynamic(il::int_t n);
  Dynamic(double x);
  Dynamic(const char* string);
  Dynamic(const il::String& string);
  Dynamic(const il::Array<il::Dynamic>& array);
  Dynamic(const il::HashMapArray<il::String, il::Dynamic>& hashmaparray);
  explicit Dynamic(il::DynamicType);
  Dynamic(const Dynamic& other);
  Dynamic(Dynamic&& other);
  Dynamic& operator=(const Dynamic& other);
  Dynamic& operator=(Dynamic&& other);
  ~Dynamic();
  bool is_null() const;
  bool is_boolean() const;
  bool is_integer() const;
  bool is_floating_point() const;
  bool is_string() const;
  bool is_hashmaparray() const;
  bool is_array() const;
  il::DynamicType type() const;
  bool to_boolean() const;
  il::int_t to_integer() const;
  double to_floating_point() const;
  il::String& as_string();
  const il::String& as_string() const;
  const il::String& as_const_string() const;
  il::Array<il::Dynamic>& as_array();
  const il::Array<il::Dynamic>& as_array() const;
  const il::Array<il::Dynamic>& as_const_array() const;
  il::HashMapArray<il::String, il::Dynamic>& as_hashmaparray();
  const il::HashMapArray<il::String, il::Dynamic>& as_hashmaparray() const;
  const il::HashMapArray<il::String, il::Dynamic>& as_const_hashmaparray()
      const;

 private:
  bool is_trivial() const;
  il::int_t& as_integer();
  const il::int_t& as_integer() const;
};

inline Dynamic::Dynamic() { data2_[3] = 0x7FF2; }

inline Dynamic::Dynamic(bool value) {
  data2_[3] = 0x7FF6;
  boolean_ = value;
}

inline Dynamic::Dynamic(int n) : Dynamic{static_cast<il::int_t>(n)} {}

inline Dynamic::Dynamic(il::int_t n) {
  const il::int_t max_integer = static_cast<il::int_t>(1) << 47;

  if (n < max_integer && n >= -max_integer) {
    // Store the integer on the stack
    n_ = n + static_cast<std::int64_t>(
                 (static_cast<std::uint64_t>(n) & 0x8000000000000000)
                     ? 0x0001000000000000
                     : 0x0);
    data2_[3] = 0x7FFA;
  } else {
    // Store the integer on the heap
    p_ = static_cast<void*>(new il::int_t{n});
    data2_[3] = 0xFFFF;
  }
}

inline Dynamic::Dynamic(double x) {
  x_ = x;
  IL_EXPECT_MEDIUM(!((data2_[3] & 0x7FF0) == 0x7FF0 && (data2_[3] & 0x0003)));
}

inline Dynamic::Dynamic(const char* string) {
  p_ = static_cast<void*>(new il::String{string});
  data2_[3] = 0x7FF1;
}

inline Dynamic::Dynamic(const il::String& string) {
  p_ = static_cast<void*>(new il::String{string});
  data2_[3] = 0x7FF1;
}

inline Dynamic::Dynamic(const il::Array<il::Dynamic>& array) {
  p_ = static_cast<void*>(new il::Array<il::Dynamic>{array});
  data2_[3] = 0x7FF3;
}

inline Dynamic::Dynamic(
    const il::HashMapArray<il::String, il::Dynamic>& hashmaparray) {
  p_ = static_cast<void*>(
      new il::HashMapArray<il::String, il::Dynamic>{hashmaparray});
  data2_[3] = 0x7FF5;
}

inline Dynamic::Dynamic(il::DynamicType type) {
  switch (type) {
    case il::DynamicType::null:
      data2_[3] = 0x7FF2;
      break;
    case il::DynamicType::boolean:
      boolean_ = false;
      data2_[3] = 0x7FF6;
      break;
    case il::DynamicType::integer:
      n_ = 0;
      data2_[3] = 0x7FFA;
      break;
    case il::DynamicType::floating_point:
      x_ = 0.0;
      break;
    case il::DynamicType::string:
      p_ = static_cast<void*>(new il::String{});
      data2_[3] = 0x7FF1;
      break;
    case il::DynamicType::array:
      p_ = static_cast<void*>(new il::Array<il::Dynamic>{});
      data2_[3] = 0x7FF3;
      break;
    case il::DynamicType::hashmaparray:
      p_ = static_cast<void*>(new il::HashMapArray<il::String, il::Dynamic>{});
      data2_[3] = 0x7FF5;
      break;
    default:
      il::abort();
  }
}

inline Dynamic::Dynamic(const Dynamic& other) {
  const bool other_trivial = other.is_trivial();
  if (other_trivial) {
    data8_ = other.data8_;
  } else {
    il::DynamicType other_type = other.type();
    switch (other_type) {
      case il::DynamicType::integer:
        p_ = static_cast<void*>(new il::int_t{other.to_integer()});
        data2_[3] = 0xFFFF;
        break;
      case il::DynamicType::string:
        p_ = static_cast<void*>(new il::String{other.as_string()});
        data2_[3] = 0x7FF1;
        break;
      case il::DynamicType::array:
        p_ = static_cast<void*>(new il::Array<il::Dynamic>{other.as_array()});
        data2_[3] = 0x7FF3;
        break;
      case il::DynamicType::hashmaparray:
        p_ = static_cast<void*>(new il::HashMapArray<il::String, il::Dynamic>{
            other.as_hashmaparray()});
        data2_[3] = 0x7FF5;
        break;
      default:
        IL_UNREACHABLE;
    }
  }
}

inline Dynamic::Dynamic(Dynamic&& other) {
  const bool other_trivial = other.is_trivial();
  if (other_trivial) {
    data8_ = other.data8_;
  } else {
    il::DynamicType other_type = other.type();
    switch (other_type) {
      case il::DynamicType::integer:
        p_ = static_cast<void*>(&other.as_integer());
        data2_[3] = 0xFFFF;
        break;
      case il::DynamicType::string:
        p_ = static_cast<void*>(&other.as_string());
        data2_[3] = 0x7FF1;
        break;
      case il::DynamicType::array:
        p_ = static_cast<void*>(&other.as_array());
        data2_[3] = 0x7FF3;
        break;
      case il::DynamicType::hashmaparray:
        p_ = static_cast<void*>(&other.as_hashmaparray());
        data2_[3] = 0x7FF5;
        break;
      default:
        IL_UNREACHABLE;
    }
  }
  other.x_ = 0.0;
}

inline il::Dynamic& Dynamic::operator=(const Dynamic& other) {
  const bool trivial = is_trivial();
  if (!trivial) {
    il::DynamicType own_type = type();
    data2_[3] &= 0x0000;
    switch (own_type) {
      case il::DynamicType::integer:
        delete static_cast<il::int_t*>(p_);
        break;
      case il::DynamicType::string:
        delete static_cast<il::String*>(p_);
        break;
      case il::DynamicType::array:
        delete static_cast<il::Array<il::Dynamic>*>(p_);
        break;
      case il::DynamicType::hashmaparray:
        delete static_cast<il::HashMapArray<il::String, il::Dynamic>*>(p_);
        break;
      default:
        IL_UNREACHABLE;
    }
  }

  const bool other_trivial = other.is_trivial();
  if (other_trivial) {
    data8_ = other.data8_;
  } else {
    il::DynamicType other_type = other.type();
    switch (other_type) {
      case il::DynamicType::integer:
        p_ = static_cast<void*>(new il::int_t{other.as_integer()});
        data2_[3] = 0xFFFF;
        break;
      case il::DynamicType::string:
        p_ = static_cast<void*>(new il::String{other.as_string()});
        data2_[3] = 0x7FF1;
        break;
      case il::DynamicType::array:
        p_ = static_cast<void*>(new il::Array<il::Dynamic>{other.as_array()});
        data2_[3] = 0x7FF3;
        break;
      case il::DynamicType::hashmaparray:
        p_ = static_cast<void*>(new il::HashMapArray<il::String, il::Dynamic>{
            other.as_hashmaparray()});
        data2_[3] = 0x7FF5;
        break;
      default:
        IL_UNREACHABLE;
    }
  }

  return *this;
}

inline il::Dynamic& Dynamic::operator=(Dynamic&& other) {
  const bool trivial = is_trivial();
  if (!trivial) {
    il::DynamicType own_type = type();
    data2_[3] &= 0x0000;
    switch (own_type) {
      case il::DynamicType::integer:
        delete static_cast<il::int_t*>(p_);
        break;
      case il::DynamicType::string:
        delete static_cast<il::String*>(p_);
        break;
      case il::DynamicType::array:
        delete static_cast<il::Array<il::Dynamic>*>(p_);
        break;
      case il::DynamicType::hashmaparray:
        delete static_cast<il::HashMapArray<il::String, il::Dynamic>*>(p_);
        break;
      default:
        IL_UNREACHABLE;
    }
  }

  const bool other_trivial = other.is_trivial();
  if (other_trivial) {
    data8_ = other.data8_;
  } else {
    il::DynamicType other_type = other.type();
    switch (other_type) {
      case il::DynamicType::integer:
        p_ = static_cast<void*>(&other.as_integer());
        data2_[3] = 0xFFFF;
        break;
      case il::DynamicType::string:
        p_ = static_cast<void*>(&other.as_string());
        data2_[3] = 0x7FF1;
        break;
      case il::DynamicType::array:
        p_ = static_cast<void*>(&other.as_array());
        data2_[3] = 0x7FF3;
        break;
      case il::DynamicType::hashmaparray:
        p_ = static_cast<void*>(&other.as_hashmaparray());
        data2_[3] = 0x7FF5;
        break;
      default:
        IL_UNREACHABLE;
    }
  }
  other.x_ = 0.0;

  return *this;
}

inline Dynamic::~Dynamic() {
  const bool trivial = is_trivial();
  if (!trivial) {
    il::DynamicType own_type = type();
    data2_[3] &= 0x0000;
    switch (own_type) {
      case il::DynamicType::integer:
        delete static_cast<il::int_t*>(p_);
        break;
      case il::DynamicType::string:
        delete static_cast<il::String*>(p_);
        break;
      case il::DynamicType::array:
        delete static_cast<il::Array<il::Dynamic>*>(p_);
        break;
      case il::DynamicType::hashmaparray:
        delete static_cast<il::HashMapArray<il::String, il::Dynamic>*>(p_);
        break;
      default:
        IL_UNREACHABLE;
    }
  }
}

inline bool Dynamic::is_null() const { return data2_[3] == 0x7FF2; }

inline bool Dynamic::is_boolean() const { return data2_[3] == 0x7FF6; }

inline bool Dynamic::is_integer() const {
  return (data2_[3] == 0x7FFA) || (data2_[3] == 0xFFFF);
}

inline bool Dynamic::is_floating_point() const {
  return !((data2_[3] & 0x7FF0) == 0x7FF0 && (data2_[3] & 0x0003));
}

inline bool Dynamic::is_string() const { return data2_[3] == 0x7FF1; }

inline bool Dynamic::is_array() const { return data2_[3] == 0x7FF3; }

inline bool Dynamic::is_hashmaparray() const { return data2_[3] == 0x7FF5; }

inline il::DynamicType Dynamic::type() const {
  if (is_floating_point()) {
    return il::DynamicType::floating_point;
  } else {
    switch (data2_[3]) {
      case 0x7FF2:
        return il::DynamicType::null;
      case 0x7FF6:
        return il::DynamicType::boolean;
      case 0x7FFA:
      case 0xFFFF:
        return il::DynamicType::integer;
      case 0x7FF1:
        return il::DynamicType::string;
      case 0x7FF3:
        return il::DynamicType::array;
      case 0x7FF5:
        return il::DynamicType::hashmaparray;
      default:
        il::abort();
        return il::DynamicType::null;
    }
  }
}

inline bool Dynamic::to_boolean() const { return boolean_; }

inline il::int_t Dynamic::to_integer() const {
  if (is_trivial()) {
    union {
      il::int_t n;
      std::size_t data8;
    };
    // We copy the bits from the internal data of the object and we zero the
    // last 16 bits where the type information was stored. We obtain
    // an unsigned 48-bit integer.
    data8 = data8_;
    data8 = data8 & 0x0000FFFFFFFFFFFF;
    // to transform that unsigned 48-bit integer to a signed 64-bit integer
    // we need to check the sign of that integer. The sign of that number is
    // given by the 48th bit which it given by: data8 & 0x0000800000000000.
    // If this bit is equal to 1, we need to substract 2^48 to the result.
    // Otherwise, the previous result was fine.
    return n - static_cast<il::int_t>(
                   (data8 & 0x0000800000000000) ? 0x0001000000000000 : 0x0);
  } else {
    union {
      il::int_t* p_integer;
      std::size_t data8;
    };
    data8 = data8_;
    data8 = data8 & 0x0000FFFFFFFFFFFF;
    return *p_integer;
  }
}

inline il::int_t& Dynamic::as_integer() {
  union {
    il::int_t* p_integer;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_integer;
}

inline const il::int_t& Dynamic::as_integer() const {
  union {
    il::int_t* p_integer;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_integer;
}

inline double Dynamic::to_floating_point() const { return x_; }

inline il::String& Dynamic::as_string() {
  union {
    il::String* p_string;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_string;
}

inline const il::String& Dynamic::as_string() const {
  union {
    il::String* p_string;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_string;
}

inline const il::String& Dynamic::as_const_string() const {
  union {
    il::String* p_string;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_string;
}

inline il::Array<il::Dynamic>& Dynamic::as_array() {
  union {
    il::Array<il::Dynamic>* p_array;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_array;
}

inline const il::Array<il::Dynamic>& Dynamic::as_array() const {
  union {
    il::Array<il::Dynamic>* p_array;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_array;
}

inline const il::Array<il::Dynamic>& Dynamic::as_const_array() const {
  union {
    il::Array<il::Dynamic>* p_array;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_array;
}

inline il::HashMapArray<il::String, il::Dynamic>& Dynamic::as_hashmaparray() {
  union {
    il::HashMapArray<il::String, il::Dynamic>* p_hashmaparray;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_hashmaparray;
}

inline const il::HashMapArray<il::String, il::Dynamic>&
Dynamic::as_hashmaparray() const {
  union {
    il::HashMapArray<il::String, il::Dynamic>* p_hashmaparray;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_hashmaparray;
}

inline const il::HashMapArray<il::String, il::Dynamic>&
Dynamic::as_const_hashmaparray() const {
  union {
    il::HashMapArray<il::String, il::Dynamic>* p_hashmaparray;
    std::size_t data8;
  };
  data8 = data8_;
  data8 = data8 & 0x0000FFFFFFFFFFFF;
  return *p_hashmaparray;
}

inline bool Dynamic::is_trivial() const {
  return !(((data2_[3] & 0x7FF0) == 0x7FF0) && (data2_[3] & 0x0001));
}
}  // namespace il

#endif  // IL_DYNAMIC_H
