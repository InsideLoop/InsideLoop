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
#include <il/Map.h>
#include <il/MapArray.h>
#include <il/String.h>

namespace il {

enum class BinaryType {
  Bool,
  Uint8,
  Int8,
  Uint16,
  Int16,
  Uint32,
  Int32,
  Uint64,
  Int64,
  Float,
  Double,
  String
};

enum class Type : unsigned char {
  Null = 0,
  Bool = 1,
  Integer = 2,
  Float = 3,
  Double = 4,
  String = 5,
  Array,
  ArrayOfUint8,
  ArrayOfFloat,
  ArrayOfDouble,
  Array2dOfUint8,
  Array2dOfFloat,
  Array2dOfDouble,
  Array2cOfUint8,
  Array2cOfFloat,
  Array2cOfDouble,
  Map,
  MapArray
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

    il::Map<il::String, il::Dynamic>* map_val_;
    il::MapArray<il::String, il::Dynamic>* map_array_val_;
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
  template <il::int_t m>
  Dynamic(const char (&value)[m]);
  Dynamic(il::StringType t, const char* value, il::int_t n);
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
  Dynamic(const il::Map<il::String, il::Dynamic>& value);
  Dynamic(il::Map<il::String, il::Dynamic>&& value);
  Dynamic(const il::MapArray<il::String, il::Dynamic>& value);
  Dynamic(il::MapArray<il::String, il::Dynamic>&& value);
  Dynamic(il::Type type);

  Dynamic(const il::Dynamic& other);
  Dynamic(il::Dynamic&& other);
  ~Dynamic();
  il::Dynamic& operator=(const il::Dynamic& other);
  il::Dynamic& operator=(il::Dynamic&& other);

  il::Type type() const;
  bool isNull() const;
  bool isBool() const;
  bool isInteger() const;
  bool isFloat() const;
  bool isDouble() const;
  bool isString() const;
  bool isArray() const;
  bool isArrayOfDouble() const;
  bool isArray2dOfDouble() const;
  bool isArray2dOfUint8() const;
  bool isMap() const;
  bool isMapArray() const;

  bool toBool() const;
  il::int_t toInteger() const;
  double toFloat() const;
  double toDouble() const;
  const il::String& asString() const;
  il::String& asString();
  const il::Array<il::Dynamic>& asArray() const;
  il::Array<il::Dynamic>& asArray();
  const il::Array<double>& asArrayOfDouble() const;
  il::Array<double>& asArrayOfDouble();
  const il::Array2D<double>& asArray2dOfDouble() const;
  il::Array2D<double>& asArray2dOfDouble();
  const il::Array2D<unsigned char>& asArray2dOfUint8() const;
  il::Array2D<unsigned char>& asArray2dOfUint8();
  const il::Map<il::String, il::Dynamic>& asMap() const;
  il::Map<il::String, il::Dynamic>& asMap();
  const il::MapArray<il::String, il::Dynamic>& asMapArray() const;
  il::MapArray<il::String, il::Dynamic>& asMapArray();

 private:
  void releaseMemory();
};

inline Dynamic::Dynamic() { type_ = il::Type::Null; }

inline Dynamic::Dynamic(bool value) {
  type_ = il::Type::Bool;
  bool_val_ = value;
}

inline Dynamic::Dynamic(int value) {
  type_ = il::Type::Integer;
  integer_val_ = value;
}

#ifdef IL_64_BIT
inline Dynamic::Dynamic(il::int_t value) {
  type_ = il::Type::Integer;
  integer_val_ = value;
}
#endif

inline Dynamic::Dynamic(float value) {
  type_ = il::Type::Float;
  float_val_ = value;
}

inline Dynamic::Dynamic(double value) {
  type_ = il::Type::Double;
  double_val_ = value;
}

template <il::int_t m>
inline Dynamic::Dynamic(const char (&value)[m]) {
  type_ = il::Type::String;
  string_val_ = new il::String{value};
}

inline Dynamic::Dynamic(il::StringType t, const char* value, il::int_t n) {
  type_ = il::Type::String;
  string_val_ = new il::String{t, value, n};
}

inline Dynamic::Dynamic(const il::String& value) {
  type_ = il::Type::String;
  string_val_ = new il::String{value};
}

inline Dynamic::Dynamic(il::String&& value) {
  type_ = il::Type::String;
  string_val_ = new il::String{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<il::Dynamic>& value) {
  type_ = il::Type::Array;
  array_val_ = new il::Array<il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Array<il::Dynamic>&& value) {
  type_ = il::Type::Array;
  array_val_ = new il::Array<il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<double>& value) {
  type_ = il::Type::ArrayOfDouble;
  array_of_double_val_ = new il::Array<double>{value};
}

inline Dynamic::Dynamic(il::Array<double>&& value) {
  type_ = il::Type::ArrayOfDouble;
  array_of_double_val_ = new il::Array<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<double>& value) {
  type_ = il::Type::Array2dOfDouble;
  array2d_of_double_val_ = new il::Array2D<double>{value};
}

inline Dynamic::Dynamic(il::Array2D<double>&& value) {
  type_ = il::Type::Array2dOfDouble;
  array2d_of_double_val_ = new il::Array2D<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<unsigned char>& value) {
  type_ = il::Type::Array2dOfUint8;
  array2d_of_uint8_val_ = new il::Array2D<unsigned char>{value};
}

inline Dynamic::Dynamic(il::Array2D<unsigned char>&& value) {
  type_ = il::Type::Array2dOfUint8;
  array2d_of_uint8_val_ = new il::Array2D<unsigned char>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Map<il::String, il::Dynamic>& value) {
  type_ = il::Type::Map;
  map_val_ = new il::Map<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Map<il::String, il::Dynamic>&& value) {
  type_ = il::Type::Map;
  map_val_ = new il::Map<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::MapArray<il::String, il::Dynamic>& value) {
  type_ = il::Type::MapArray;
  map_array_val_ = new il::MapArray<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::MapArray<il::String, il::Dynamic>&& value) {
  type_ = il::Type::MapArray;
  map_array_val_ = new il::MapArray<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(il::Type value) {
  type_ = value;
  switch (value) {
    case il::Type::Map:
      map_val_ = new il::Map<il::String, il::Dynamic>{};
      break;
    case il::Type::MapArray:
      map_array_val_ = new il::MapArray<il::String, il::Dynamic>{};
      break;
    default:
      IL_UNREACHABLE;
  }
}

inline Dynamic::~Dynamic() { releaseMemory(); }

inline Dynamic::Dynamic(const il::Dynamic& other) {
  type_ = other.type_;
  if (type_ == il::Type::String) {
    string_val_ = new il::String{*other.string_val_};
  } else if (type_ == il::Type::Array) {
    array_val_ = new il::Array<il::Dynamic>{*other.array_val_};
  } else if (type_ == il::Type::ArrayOfDouble) {
    array_of_double_val_ = new il::Array<double>{*other.array_of_double_val_};
  } else if (type_ == il::Type::Array2dOfDouble) {
    array2d_of_double_val_ =
        new il::Array2D<double>{*other.array2d_of_double_val_};
  } else if (type_ == il::Type::Map) {
    map_val_ = new il::Map<il::String, il::Dynamic>{*other.map_val_};
  } else if (type_ == il::Type::MapArray) {
    map_array_val_ =
        new il::MapArray<il::String, il::Dynamic>{*other.map_array_val_};
  } else {
    data_ = other.data_;
  }
}

inline Dynamic::Dynamic(il::Dynamic&& other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::Null;
}

inline il::Dynamic& Dynamic::operator=(const il::Dynamic& other) {
  releaseMemory();
  type_ = other.type_;
  if (type_ == il::Type::String) {
    string_val_ = new il::String{*other.string_val_};
  } else if (type_ == il::Type::Array) {
    array_val_ = new il::Array<il::Dynamic>{*other.array_val_};
  } else if (type_ == il::Type::ArrayOfDouble) {
    array_of_double_val_ = new il::Array<double>{*other.array_of_double_val_};
  } else if (type_ == il::Type::Array2dOfDouble) {
    array2d_of_double_val_ =
        new il::Array2D<double>{*other.array2d_of_double_val_};
  } else if (type_ == il::Type::Map) {
    map_val_ = new il::Map<il::String, il::Dynamic>{*other.map_val_};
  } else if (type_ == il::Type::MapArray) {
    map_array_val_ =
        new il::MapArray<il::String, il::Dynamic>{*other.map_array_val_};
  } else {
    data_ = other.data_;
  }
  return *this;
}

inline il::Dynamic& Dynamic::operator=(il::Dynamic&& other) {
  releaseMemory();
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::Null;
  return *this;
}

inline il::Type Dynamic::type() const { return type_; }

inline bool Dynamic::isNull() const { return type_ == il::Type::Null; }

inline bool Dynamic::isBool() const { return type_ == il::Type::Bool; }

inline bool Dynamic::isInteger() const { return type_ == il::Type::Integer; }

inline bool Dynamic::isFloat() const { return type_ == il::Type::Float; }

inline bool Dynamic::isDouble() const { return type_ == il::Type::Double; }

inline bool Dynamic::isString() const { return type_ == il::Type::String; }

inline bool Dynamic::isArray() const { return type_ == il::Type::Array; }

inline bool Dynamic::isArrayOfDouble() const {
  return type_ == il::Type::ArrayOfDouble;
}

inline bool Dynamic::isArray2dOfDouble() const {
  return type_ == il::Type::Array2dOfDouble;
}

inline bool Dynamic::isArray2dOfUint8() const {
  return type_ == il::Type::Array2dOfUint8;
}

inline bool Dynamic::isMap() const { return type_ == il::Type::Map; }

inline bool Dynamic::isMapArray() const { return type_ == il::Type::MapArray; }

inline bool Dynamic::toBool() const { return bool_val_; }

inline il::int_t Dynamic::toInteger() const { return integer_val_; }

inline double Dynamic::toFloat() const { return float_val_; }

inline double Dynamic::toDouble() const { return double_val_; }

inline const il::String& Dynamic::asString() const { return *string_val_; }

inline il::String& Dynamic::asString() { return *string_val_; }

inline const il::Array<il::Dynamic>& Dynamic::asArray() const {
  return *array_val_;
}

inline il::Array<il::Dynamic>& Dynamic::asArray() { return *array_val_; }

inline const il::Array<double>& Dynamic::asArrayOfDouble() const {
  return *array_of_double_val_;
}

inline il::Array<double>& Dynamic::asArrayOfDouble() {
  return *array_of_double_val_;
}

inline const il::Array2D<double>& Dynamic::asArray2dOfDouble() const {
  return *array2d_of_double_val_;
}

inline il::Array2D<double>& Dynamic::asArray2dOfDouble() {
  return *array2d_of_double_val_;
}

inline const il::Array2D<unsigned char>& Dynamic::asArray2dOfUint8() const {
  return *array2d_of_uint8_val_;
}

inline il::Array2D<unsigned char>& Dynamic::asArray2dOfUint8() {
  return *array2d_of_uint8_val_;
}

inline const il::Map<il::String, il::Dynamic>& Dynamic::asMap() const {
  return *map_val_;
}

inline il::Map<il::String, il::Dynamic>& Dynamic::asMap() { return *map_val_; }

inline const il::MapArray<il::String, il::Dynamic>& Dynamic::asMapArray()
    const {
  return *map_array_val_;
}

inline il::MapArray<il::String, il::Dynamic>& Dynamic::asMapArray() {
  return *map_array_val_;
}

inline void Dynamic::releaseMemory() {
  if (type_ == il::Type::String) {
    delete string_val_;
  } else if (type_ == il::Type::Array) {
    delete array_val_;
  } else if (type_ == il::Type::ArrayOfDouble) {
    delete array_of_double_val_;
  } else if (type_ == il::Type::Array2dOfDouble) {
    delete array2d_of_double_val_;
  } else if (type_ == il::Type::Array2dOfUint8) {
    delete array2d_of_uint8_val_;
  } else if (type_ == il::Type::Map) {
    delete map_val_;
  } else if (type_ == il::Type::MapArray) {
    delete map_array_val_;
  }
}

}  // namespace il

#endif  // IL_DYN_H
