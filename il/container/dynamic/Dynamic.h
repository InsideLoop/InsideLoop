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
#include <il/Array2D.h>
#include <il/HashMapArray.h>
#include <il/String.h>

namespace il {

enum class BinaryType {
  bool_t,
  uint8_t,
  int8_t,
  uint16_t,
  int16_t,
  uint32_t,
  int32_t,
  uint64_t,
  int64_t,
  float_t,
  double_t,
  string_t
};

enum class Type : unsigned char {
  null_t = 0,
  bool_t = 1,
  integer_t = 2,
  float_t = 3,
  double_t = 4,
  string_t = 5,
  array_t,
  array_of_uint8_t,
  array_of_float_t,
  array_of_double_t,
  array2d_of_uint8_t,
  array2d_of_float_t,
  array2d_of_double_t,
  array2c_of_uint8_t,
  array2c_of_float_t,
  array2c_of_double_t,
  hash_map_t,
  hash_map_array_t
};

class Dynamic {
 private:
  il::Type type_;
  union {
    std::uint64_t data_;

    bool bool_val_;
    il::int_t integer_val_;
    float float_val_;
    double double_val_;

    il::String* string_val_;

    il::Array<il::Dynamic>* array_val_;
    il::Array<unsigned char>* array_of_uint8_val_;
    il::Array<double>* array_of_double_val_;
    il::Array2D<unsigned char>* array2d_of_uint8_val_;
    il::Array2D<double>* array2d_of_double_val_;

    il::HashMap<il::String, il::Dynamic>* hash_map_val_;
    il::HashMapArray<il::String, il::Dynamic>* hash_map_array_val_;
  };

 public:
  Dynamic();
  Dynamic(bool value);
  Dynamic(int value);
#ifdef IL_64_BIT
  Dynamic(il::int_t value);
#endif
  Dynamic(float value);
  Dynamic(double value);
  Dynamic(const char* value);
  Dynamic(const il::String& value);
  Dynamic(il::String&& value);
  Dynamic(const il::Array<il::Dynamic>& value);
  Dynamic(il::Array<il::Dynamic>&& value);
  Dynamic(const il::Array<double>& value);
  Dynamic(il::Array<double>&& value);
  Dynamic(const il::Array2D<double>& value);
  Dynamic(il::Array2D<double>&& value);
  Dynamic(const il::Array2D<unsigned char>& value);
  Dynamic(il::Array2D<unsigned char>&& value);
  Dynamic(const il::HashMap<il::String, il::Dynamic>& value);
  Dynamic(il::HashMap<il::String, il::Dynamic>&& value);
  Dynamic(const il::HashMapArray<il::String, il::Dynamic>& value);
  Dynamic(il::HashMapArray<il::String, il::Dynamic>&& value);
  Dynamic(il::Type type);

  Dynamic(const il::Dynamic& other);
  Dynamic(il::Dynamic&& other);
  ~Dynamic();
  il::Dynamic& operator=(const il::Dynamic& other);
  il::Dynamic& operator=(il::Dynamic&& other);

  il::Type type() const;
  bool is_null() const;
  bool is_bool() const;
  bool is_integer() const;
  bool is_float() const;
  bool is_double() const;
  bool is_string() const;
  bool is_array() const;
  bool is_array_of_double() const;
  bool is_array2d_of_double() const;
  bool is_array2d_of_uint8() const;
  bool is_hash_map() const;
  bool is_hash_map_array() const;

  bool to_bool() const;
  il::int_t to_integer() const;
  double to_float() const;
  double to_double() const;
  const il::String& as_string() const;
  const il::Array<il::Dynamic>& as_array() const;
  il::Array<il::Dynamic>& as_array();
  const il::Array<il::Dynamic>& as_const_array() const;
  const il::Array<double>& as_array_of_double() const;
  const il::Array2D<double>& as_array2d_of_double() const;
  const il::Array2D<unsigned char>& as_array2d_of_uint8() const;
  const il::HashMap<il::String, il::Dynamic>& as_hash_map() const;
  il::HashMap<il::String, il::Dynamic>& as_hash_map();
  const il::HashMapArray<il::String, il::Dynamic>& as_hash_map_array() const;
  const il::HashMapArray<il::String, il::Dynamic>& as_const_hash_map_array() const;
  il::HashMapArray<il::String, il::Dynamic>& as_hash_map_array();

 private:
  void release_memory();
};

inline Dynamic::Dynamic() { type_ = il::Type::null_t; }

inline Dynamic::Dynamic(bool value) {
  type_ = il::Type::bool_t;
  bool_val_ = value;
}

inline Dynamic::Dynamic(int value) {
  type_ = il::Type::integer_t;
  integer_val_ = value;
}

#ifdef IL_64_BIT
inline Dynamic::Dynamic(il::int_t value) {
  type_ = il::Type::integer_t;
  integer_val_ = value;
}
#endif

inline Dynamic::Dynamic(float value) {
  type_ = il::Type::float_t;
  float_val_ = value;
}

inline Dynamic::Dynamic(double value) {
  type_ = il::Type::double_t;
  double_val_ = value;
}

inline Dynamic::Dynamic(const char* value) {
  type_ = il::Type::string_t;
  string_val_ = new il::String{value};
}

inline Dynamic::Dynamic(const il::String& value) {
  type_ = il::Type::string_t;
  string_val_ = new il::String{value};
}

inline Dynamic::Dynamic(il::String&& value) {
  type_ = il::Type::string_t;
  string_val_ = new il::String{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<il::Dynamic>& value) {
  type_ = il::Type::array_t;
  array_val_ = new il::Array<il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Array<il::Dynamic>&& value) {
  type_ = il::Type::array_t;
  array_val_ = new il::Array<il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<double>& value) {
  type_ = il::Type::array_of_double_t;
  array_of_double_val_ = new il::Array<double>{value};
}

inline Dynamic::Dynamic(il::Array<double>&& value) {
  type_ = il::Type::array_of_double_t;
  array_of_double_val_ = new il::Array<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<double>& value) {
  type_ = il::Type::array2d_of_double_t;
  array2d_of_double_val_ = new il::Array2D<double>{value};
}

inline Dynamic::Dynamic(il::Array2D<double>&& value) {
  type_ = il::Type::array2d_of_double_t;
  array2d_of_double_val_ = new il::Array2D<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<unsigned char>& value) {
  type_ = il::Type::array2d_of_uint8_t;
  array2d_of_uint8_val_ = new il::Array2D<unsigned char>{value};
}

inline Dynamic::Dynamic(il::Array2D<unsigned char>&& value) {
  type_ = il::Type::array2d_of_uint8_t;
  array2d_of_uint8_val_ = new il::Array2D<unsigned char>{std::move(value)};
}

inline Dynamic::Dynamic(const il::HashMap<il::String, il::Dynamic>& value) {
  type_ = il::Type::hash_map_t;
  hash_map_val_ = new il::HashMap<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::HashMap<il::String, il::Dynamic>&& value) {
  type_ = il::Type::hash_map_t;
  hash_map_val_ = new il::HashMap<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(
    const il::HashMapArray<il::String, il::Dynamic>& value) {
  type_ = il::Type::hash_map_array_t;
  hash_map_array_val_ = new il::HashMapArray<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::HashMapArray<il::String, il::Dynamic>&& value) {
  type_ = il::Type::hash_map_array_t;
  hash_map_array_val_ =
      new il::HashMapArray<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(il::Type value) {
  type_ = value;
  switch (value) {
    case il::Type::hash_map_t:
      hash_map_val_ = new il::HashMap<il::String, il::Dynamic>{};
      break;
    case il::Type::hash_map_array_t:
      hash_map_array_val_ = new il::HashMapArray<il::String, il::Dynamic>{};
      break;
    default:
      IL_UNREACHABLE;
  }
}

inline Dynamic::~Dynamic() { release_memory(); }

inline Dynamic::Dynamic(const il::Dynamic& other) {
  type_ = other.type_;
  if (type_ == il::Type::string_t) {
    string_val_ = new il::String{*other.string_val_};
  } else if (type_ == il::Type::array_t) {
    array_val_ = new il::Array<il::Dynamic>{*other.array_val_};
  } else if (type_ == il::Type::array_of_double_t) {
    array_of_double_val_ = new il::Array<double>{*other.array_of_double_val_};
  } else if (type_ == il::Type::array2d_of_double_t) {
    array2d_of_double_val_ =
        new il::Array2D<double>{*other.array2d_of_double_val_};
  } else if (type_ == il::Type::hash_map_t) {
    hash_map_val_ =
        new il::HashMap<il::String, il::Dynamic>{*other.hash_map_val_};
  } else if (type_ == il::Type::hash_map_array_t) {
    hash_map_array_val_ = new il::HashMapArray<il::String, il::Dynamic>{
        *other.hash_map_array_val_};
  } else {
    data_ = other.data_;
  }
}

inline Dynamic::Dynamic(il::Dynamic&& other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::null_t;
}

inline il::Dynamic& Dynamic::operator=(const il::Dynamic& other) {
  release_memory();
  type_ = other.type_;
  if (type_ == il::Type::string_t) {
    string_val_ = new il::String{*other.string_val_};
  } else if (type_ == il::Type::array_t) {
    array_val_ = new il::Array<il::Dynamic>{*other.array_val_};
  } else if (type_ == il::Type::array_of_double_t) {
    array_of_double_val_ = new il::Array<double>{*other.array_of_double_val_};
  } else if (type_ == il::Type::array2d_of_double_t) {
    array2d_of_double_val_ =
        new il::Array2D<double>{*other.array2d_of_double_val_};
  } else if (type_ == il::Type::hash_map_t) {
    hash_map_val_ =
        new il::HashMap<il::String, il::Dynamic>{*other.hash_map_val_};
  } else if (type_ == il::Type::hash_map_array_t) {
    hash_map_array_val_ = new il::HashMapArray<il::String, il::Dynamic>{
        *other.hash_map_array_val_};
  } else {
    data_ = other.data_;
  }
  return *this;
}

inline il::Dynamic& Dynamic::operator=(il::Dynamic&& other) {
  release_memory();
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::null_t;
  return *this;
}

inline il::Type Dynamic::type() const { return type_; }

inline bool Dynamic::is_null() const { return type_ == il::Type::null_t; }

inline bool Dynamic::is_bool() const { return type_ == il::Type::bool_t; }

inline bool Dynamic::is_integer() const { return type_ == il::Type::integer_t; }

inline bool Dynamic::is_float() const { return type_ == il::Type::float_t; }

inline bool Dynamic::is_double() const { return type_ == il::Type::double_t; }

inline bool Dynamic::is_string() const { return type_ == il::Type::string_t; }

inline bool Dynamic::is_array() const { return type_ == il::Type::array_t; }

inline bool Dynamic::is_array_of_double() const {
  return type_ == il::Type::array_of_double_t;
}

inline bool Dynamic::is_array2d_of_double() const {
  return type_ == il::Type::array2d_of_double_t;
}

inline bool Dynamic::is_array2d_of_uint8() const {
  return type_ == il::Type::array2d_of_uint8_t;
}

inline bool Dynamic::is_hash_map() const {
  return type_ == il::Type::hash_map_t;
}

inline bool Dynamic::is_hash_map_array() const {
  return type_ == il::Type::hash_map_array_t;
}

inline bool Dynamic::to_bool() const { return bool_val_; }

inline il::int_t Dynamic::to_integer() const { return integer_val_; }

inline double Dynamic::to_float() const { return float_val_; }

inline double Dynamic::to_double() const { return double_val_; }

inline const il::String& Dynamic::as_string() const { return *string_val_; }

inline const il::Array<il::Dynamic>& Dynamic::as_array() const {
  return *array_val_;
}

inline il::Array<il::Dynamic>& Dynamic::as_array() { return *array_val_; }

inline const il::Array<il::Dynamic>& Dynamic::as_const_array() const {
  return *array_val_;
}

inline const il::Array<double>& Dynamic::as_array_of_double() const {
  return *array_of_double_val_;
}

inline const il::Array2D<double>& Dynamic::as_array2d_of_double() const {
  return *array2d_of_double_val_;
}

inline const il::Array2D<unsigned char>& Dynamic::as_array2d_of_uint8() const {
  return *array2d_of_uint8_val_;
}

inline const il::HashMap<il::String, il::Dynamic>& Dynamic::as_hash_map()
    const {
  return *hash_map_val_;
}

inline il::HashMap<il::String, il::Dynamic>& Dynamic::as_hash_map() {
  return *hash_map_val_;
}

inline const il::HashMapArray<il::String, il::Dynamic>&
Dynamic::as_hash_map_array() const {
  return *hash_map_array_val_;
}

inline const il::HashMapArray<il::String, il::Dynamic>&
Dynamic::as_const_hash_map_array() const {
  return *hash_map_array_val_;
}

inline il::HashMapArray<il::String, il::Dynamic>& Dynamic::as_hash_map_array() {
  return *hash_map_array_val_;
}

inline void Dynamic::release_memory() {
  if (type_ == il::Type::string_t) {
    delete string_val_;
  } else if (type_ == il::Type::array_t) {
    delete array_val_;
  } else if (type_ == il::Type::array_of_double_t) {
    delete array_of_double_val_;
  } else if (type_ == il::Type::array2d_of_double_t) {
    delete array2d_of_double_val_;
  } else if (type_ == il::Type::array2d_of_uint8_t) {
    delete array2d_of_uint8_val_;
  } else if (type_ == il::Type::hash_map_t) {
    delete hash_map_val_;
  } else if (type_ == il::Type::hash_map_array_t) {
    delete hash_map_array_val_;
  }
}

}  // namespace il

#endif  // IL_DYN_H
