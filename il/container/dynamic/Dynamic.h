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
  union {
    unsigned char data_[8];
    il::int_t n_;
    double x_;
    il::String* p_string_;
    il::Array<il::Dynamic>* p_array_;
    il::HashMap<il::String, il::Dynamic>* p_hashmap_;
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
  Dynamic(const il::HashMap<il::String, il::Dynamic>& hashmap);
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
  bool is_hashmap() const;
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
  p_string_ = new il::String{string};
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
  data_[6] = 0xF0 | 0x05;
}

inline Dynamic::Dynamic(il::DynamicType type) {
  switch (type) {
    case il::DynamicType::array: {
      p_array_ = new il::Array<il::Dynamic>{};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x04;
    } break;
    case il::DynamicType::hashmap: {
      p_hashmap_ = new il::HashMap<il::String, il::Dynamic>{};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x05;
    } break;
    default:
      il::abort();
  }
}

inline Dynamic::Dynamic(const Dynamic& other) {
  il::DynamicType other_type = other.type();

  switch (other_type) {
    case il::DynamicType::boolean: {
      data_[7] = 0x80;
      if (other.to_boolean()) {
        data_[6] = 0xF0 | 0x07;
      } else {
        data_[6] = 0xF0 | 0x06;
      }
    } break;
    case il::DynamicType::integer: {
      const il::int_t n = other.to_integer();
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
      *reinterpret_cast<double*>(data_) = other.to_floating_point();
    } break;
    case il::DynamicType::string: {
      p_string_ = new il::String{other.as_string()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x03;
    } break;
    case il::DynamicType::array: {
      p_array_ = new il::Array<il::Dynamic>{other.as_array()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x04;
    } break;
    case il::DynamicType::hashmap: {
      p_hashmap_ = new il::HashMap<il::String, il::Dynamic>{other.as_hashmap()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x05;
    } break;
    default:
      il::abort();
  }
}

inline Dynamic::Dynamic(Dynamic&& other) {
  switch (other.type()) {
    case il::DynamicType::null:
    case il::DynamicType::boolean:
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
      std::memcpy(data_, other.data_, 8);
      break;
    case il::DynamicType::string: {
      p_string_ = &other.as_string();
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x03;
    } break;
    case il::DynamicType::array: {
      p_array_ = &other.as_array();
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x04;
    } break;
    case il::DynamicType::hashmap: {
      p_hashmap_ = &other.as_hashmap();
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x05;
    } break;
    default:
      il::abort();
  }

  other.x_ = 0.0;
}

inline il::Dynamic& Dynamic::operator=(const Dynamic& other) {
  switch (type()) {
    case il::DynamicType::null:
    case il::DynamicType::boolean:
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
      break;
    case il::DynamicType::string: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_string_;
    } break;
    case il::DynamicType::array: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_array_;
    } break;
    case il::DynamicType::hashmap: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_hashmap_;
    } break;
    default:
      il::abort();
  }

  switch (other.type()) {
    case il::DynamicType::null:
    case il::DynamicType::boolean:
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
      std::memcpy(data_, other.data_, 8);
      break;
    case il::DynamicType::string:
      p_string_ = new il::String{other.as_string()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x03;
      break;
    case il::DynamicType::array:
      p_array_ = new il::Array<il::Dynamic>{other.as_array()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x04;
      break;
    case il::DynamicType::hashmap:
      p_hashmap_ = new il::HashMap<il::String, il::Dynamic>{other.as_hashmap()};
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x05;
      break;
    default:
      il::abort();
  }

  return *this;
}

inline il::Dynamic& Dynamic::operator=(Dynamic&& other) {
  switch (type()) {
    case il::DynamicType::null:
    case il::DynamicType::boolean:
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
      break;
    case il::DynamicType::string: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_string_;
    } break;
    case il::DynamicType::array: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_array_;
    } break;
    case il::DynamicType::hashmap: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_hashmap_;
    } break;
    default:
      il::abort();
  }

  switch (other.type()) {
    case il::DynamicType::null:
    case il::DynamicType::boolean:
    case il::DynamicType::integer:
    case il::DynamicType::floating_point:
      std::memcpy(data_, other.data_, 8);
      break;
    case il::DynamicType::string:
      p_string_ = other.p_string_;
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x03;
      break;
    case il::DynamicType::array:
      p_array_ = other.p_array_;
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x04;
      break;
    case il::DynamicType::hashmap:
      p_hashmap_ = other.p_hashmap_;
      data_[7] = 0x80;
      data_[6] = 0xF0 | 0x05;
      break;
    default:
      il::abort();
  }

  other.x_ = 0.0;
  return *this;
}

inline Dynamic::~Dynamic() {
  switch (type()) {
    case il::DynamicType::string: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_string_;
    } break;
    case il::DynamicType::array: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_array_;
    } break;
    case il::DynamicType::hashmap: {
      data_[6] = 0x00;
      data_[7] = 0x00;
      delete p_hashmap_;
    } break;
    default:
      break;
  }
}

inline bool Dynamic::is_null() const { return type() == il::DynamicType::null; }

inline bool Dynamic::is_boolean() const {
  return type() == il::DynamicType::boolean;
}

inline bool Dynamic::is_integer() const {
  return type() == il::DynamicType::integer;
}

inline bool Dynamic::is_floating_point() const {
  return type() == il::DynamicType::floating_point;
}

inline bool Dynamic::is_string() const {
  return type() == il::DynamicType::string;
}

inline bool Dynamic::is_hashmap() const {
  return type() == il::DynamicType::hashmap;
}

inline bool Dynamic::is_array() const {
  return type() == il::DynamicType::array;
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

inline bool Dynamic::to_boolean() const {
  return ((data_[6] & 0x07) == 0x06) ? false : true;
}

inline il::int_t Dynamic::to_integer() const {
  unsigned char data_local[8];
  std::memcpy(data_local, data_, 8);
  data_local[6] = 0x00;
  data_local[7] = 0x00;
  il::int_t n = *reinterpret_cast<il::int_t*>(data_local);
  const il::int_t max_integer = static_cast<il::int_t>(1) << 47;
  return (n < max_integer) ? n : n - 2 * max_integer;
}

inline double Dynamic::to_floating_point() const {
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
