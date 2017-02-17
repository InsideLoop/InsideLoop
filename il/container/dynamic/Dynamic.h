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
#include <il/HashMap.h>
#include <il/String.h>

namespace il {

enum class DynamicType {
  null,
  boolean,
  integer,
  floating_point,
  string,
  array,
  hashmap
};

class Dynamic {
 private:
  unsigned char data_[sizeof(double)];

 public:
  Dynamic();
  Dynamic(bool value);
  Dynamic(int n);
  Dynamic(il::int_t n);
  Dynamic(double x);
  Dynamic(const char* string);
  Dynamic(const il::String& string);
  Dynamic(const il::Array<il::Dynamic>& array);
  Dynamic(const il::HashMap<il::String, il::Dynamic>& hashmap);
  explicit Dynamic(il::DynamicType);
  Dynamic(const Dynamic& other);
  ~Dynamic();
  il::DynamicType type() const;
  bool get_boolean() const;
  il::int_t get_integer() const;
  double get_floating_point() const;
  il::String& as_string();
  const il::String& as_string() const;
  const il::String& as_const_string() const;
  il::Array<il::Dynamic>& as_array();
  const il::Array<il::Dynamic>& as_array() const;
  const il::Array<il::Dynamic>& as_const_array() const;
  il::HashMap<il::String, il::Dynamic>& as_hashmap();
  const il::HashMap<il::String, il::Dynamic>& as_hashmap() const;
  const il::HashMap<il::String, il::Dynamic>& as_const_hashmap() const;
};

inline Dynamic::Dynamic() {
  data_[7] = 0x80;
  data_[6] = 0xF0 | 0x01;
}

inline Dynamic::Dynamic(bool value) {
  data_[7] = 0x80;
  if (value) {
    data_[6] = 0xF0 | 0x07;
  } else {
    data_[6] = 0xF0 | 0x06;
  }
}

inline Dynamic::Dynamic(int n) : Dynamic{static_cast<il::int_t>(n)} {}

inline Dynamic::Dynamic(il::int_t n) {
  const il::int_t max_integer = static_cast<il::int_t>(1) << 47;
  IL_EXPECT_MEDIUM((n >= 0 && n < max_integer) || (n < 0 && n >= -max_integer));

  if (n >= 0) {
    *reinterpret_cast<il::int_t*>(data_) = n;
  } else {
    std::size_t n_unsigned = n;
    n_unsigned += static_cast<std::size_t>(1) << 48;
    *reinterpret_cast<std::size_t*>(data_) = n;
  }
  data_[7] = 0x80;
  data_[6] = 0xF0 | 0x02;
}

inline Dynamic::Dynamic(double x) { *reinterpret_cast<double*>(data_) = x; }

inline Dynamic::Dynamic(const char* string) {
  il::String** p = reinterpret_cast<il::String**>(data_);
  il::String* pointer = new il::String{string};
  *p = pointer;
  data_[7] = 0x80;
  data_[6] = 0xF0 | 0x03;
}

inline Dynamic::Dynamic(const il::String& string) {
  il::String** p = reinterpret_cast<il::String**>(data_);
  *p = new il::String{string};
  data_[7] = 0x80;
  data_[6] = 0xF0 | 0x03;
}

inline Dynamic::Dynamic(const il::Array<il::Dynamic>& array) {
  il::Array<il::Dynamic>** p =
      reinterpret_cast<il::Array<il::Dynamic>**>(data_);
  *p = new il::Array<il::Dynamic>{array};
  data_[7] = 0x80;
  data_[6] = 0xF0 | 0x04;
}

inline Dynamic::Dynamic(const il::HashMap<il::String, il::Dynamic>& hashmap) {
  il::HashMap<il::String, il::Dynamic>** p =
      reinterpret_cast<il::HashMap<il::String, il::Dynamic>**>(data_);
  *p = new il::HashMap<il::String, il::Dynamic>{hashmap};
  data_[7] = 0x80;
  data_[6] = 0xF0 | 0x04;
}

inline Dynamic::Dynamic(il::DynamicType type) {
  switch (type) {
    case il::DynamicType::hashmap: {
      il::HashMap<il::String, il::Dynamic> **p =
          reinterpret_cast<il::HashMap<il::String, il::Dynamic> **>(data_);
      *p = new il::HashMap<il::String, il::Dynamic>{};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x04;
    } break;
    default:
    il::abort();
  }
}

inline Dynamic::Dynamic(const Dynamic& other) {
  il::DynamicType type = other.type();

  switch (type) {
    case il::DynamicType::boolean: {
      data_[7] = 0x80;
      if (other.get_boolean()) {
        data_[6] = 0xF0 | 0x07;
      } else {
        data_[6] = 0xF0 | 0x06;
      }
    } break;
    case il::DynamicType::integer: {
      const il::int_t n = other.get_integer();
      if (n >= 0) {
        *reinterpret_cast<il::int_t*>(data_) = n;
      } else {
        std::size_t n_unsigned = n;
        n_unsigned += static_cast<std::size_t>(1) << 48;
        *reinterpret_cast<std::size_t*>(data_) = n;
      }
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x02;
    } break;
    case il::DynamicType::floating_point: {
      *reinterpret_cast<double*>(data_) = other.get_floating_point();
    } break;
    case il::DynamicType::string: {
      il::String** p = reinterpret_cast<il::String**>(data_);
      *p = new il::String{other.as_const_string()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x03;
    } break;
    default:
      il::abort();
  }
}

inline Dynamic::~Dynamic() {
  switch (type()) {
    case il::DynamicType::string: {
      union {
        unsigned char data_local[8];
        il::String* p;
      };
      std::memcpy(data_local, data_, 8);
      data_local[6] = 0x00;
      data_local[7] = 0x00;
      delete p;
    } break;
    case il::DynamicType::array: {
      unsigned char data_local[8];
      std::memcpy(data_local, data_, 8);
      data_local[6] = 0x00;
      data_local[7] = 0x00;
      delete reinterpret_cast<il::Array<il::Dynamic>*>(data_local);
    } break;
    case il::DynamicType::hashmap: {
      unsigned char data_local[8];
      std::memcpy(data_local, data_, 8);
      data_local[6] = 0x00;
      data_local[7] = 0x00;
      delete reinterpret_cast<il::HashMap<il::String, il::Dynamic>*>(
          data_local);
    } break;
    default:
      break;
  }
}

inline il::DynamicType Dynamic::type() const {
  bool exponent_filled =
      ((data_[7] & 0x80) == 0x80) && ((data_[6] & 0xF0) == 0xF0);
  unsigned char encoded_type = data_[6] & 0x07;
  if (!exponent_filled || (encoded_type == 0x00)) {
    return il::DynamicType::floating_point;
  } else {
    switch (encoded_type) {
      case 0x01:
        return il::DynamicType::null;
      case 0x02:
        return il::DynamicType::integer;
      case 0x03:
        return il::DynamicType::string;
      case 0x04:
        return il::DynamicType::array;
      case 0x05:
        return il::DynamicType::hashmap;
      case 0x06:
      case 0x07:
        return il::DynamicType::boolean;
      default:
        IL_UNREACHABLE;
        return il::DynamicType::null;
    }
  }
}

inline bool Dynamic::get_boolean() const {
  return ((data_[6] & 0x07) == 0x06) ? false : true;
}

inline il::int_t Dynamic::get_integer() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::int_t n = *reinterpret_cast<il::int_t*>(data_local);
  const il::int_t max_integer = static_cast<il::int_t>(1) << 47;
  return (n < max_integer) ? n : n - 2 * max_integer;
}

inline double Dynamic::get_floating_point() const {
  return *reinterpret_cast<const double*>(data_);
}

inline il::String& Dynamic::as_string() {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::String** p = reinterpret_cast<il::String**>(data_local);
  return **p;
}

inline const il::String& Dynamic::as_string() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::String** p = reinterpret_cast<il::String**>(data_local);
  return **p;
}

inline const il::String& Dynamic::as_const_string() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::String** p = reinterpret_cast<il::String**>(data_local);
  return **p;
}

inline il::Array<il::Dynamic>& Dynamic::as_array() {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::Array<il::Dynamic>** p =
      reinterpret_cast<il::Array<il::Dynamic>**>(data_local);
  return **p;
}

inline const il::Array<il::Dynamic>& Dynamic::as_array() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::Array<il::Dynamic>** p =
      reinterpret_cast<il::Array<il::Dynamic>**>(data_local);
  return **p;
}

inline const il::Array<il::Dynamic>& Dynamic::as_const_array() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::Array<il::Dynamic>** p =
      reinterpret_cast<il::Array<il::Dynamic>**>(data_local);
  return **p;
}

inline il::HashMap<il::String, il::Dynamic>& Dynamic::as_hashmap() {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::HashMap<il::String, il::Dynamic>** p =
      reinterpret_cast<il::HashMap<il::String, il::Dynamic>**>(data_local);
  return **p;
}

inline const il::HashMap<il::String, il::Dynamic>& Dynamic::as_hashmap() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::HashMap<il::String, il::Dynamic>** p =
      reinterpret_cast<il::HashMap<il::String, il::Dynamic>**>(data_local);
  return **p;
}

inline const il::HashMap<il::String, il::Dynamic>& Dynamic::as_const_hashmap()
    const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::HashMap<il::String, il::Dynamic>** p =
      reinterpret_cast<il::HashMap<il::String, il::Dynamic>**>(data_local);
  return **p;
}
}

#endif  // IL_DYNAMIC_H
