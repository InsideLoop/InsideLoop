//==============================================================================
//
// Copyright 2017 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

enum class Type : unsigned char {
  TNull,
  TBool,
  TInteger,
  TFloat,
  TDouble,
  TString,
  TArray,
  TArrayOfUint8,
  TArrayOfInt8,
  TArrayOfUint16,
  TArrayOfInt16,
  TArrayOfUint32,
  TArrayOfInt32,
  TArrayOfInteger,
  TArrayOfUinteger,
  TArrayOfFloat,
  TArrayOfDouble,
  TArray2dOfUint8,
  TArray2dOfInt8,
  TArray2dOfUint16,
  TArray2dOfInt16,
  TArray2dOfUint32,
  TArray2dOfInt32,
  TArray2dOfInteger,
  TArray2dOfUinteger,
  TArray2dOfFloat,
  TArray2dOfDouble,
  TArray2cOfUint8,
  TArray2cOfInt8,
  TArray2cOfUint16,
  TArray2cOfInt16,
  TArray2cOfUint32,
  TArray2cOfInt32,
  TArray2cOfInteger,
  TArray2cOfUinteger,
  TArray2cOfFloat,
  TArray2cOfDouble,
  TMap,
  TMapArray,
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
    il::Array<int>* array_of_int32_val_;
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
  Dynamic(const il::Array<unsigned char>& value);
  Dynamic(il::Array<unsigned char>&& value);
  Dynamic(const il::Array<int>& value);
  Dynamic(il::Array<int>&& value);
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
  bool isArrayOfUint8() const;
  bool isArrayOfInt32() const;
  bool isArrayOfDouble() const;
  bool isArray2dOfDouble() const;
  bool isArray2dOfUint8() const;
  bool isMap() const;
  bool isMapArray() const;

  bool toBool() const;
  il::int_t toInteger() const;
  float toFloat() const;
  double toDouble() const;
  const il::String& asString() const;
  il::String& asString();
  const il::Array<il::Dynamic>& asArray() const;
  il::Array<il::Dynamic>& asArray();
  const il::Array<unsigned char>& asArrayOfUint8() const;
  il::Array<unsigned char>& asArrayOfUint8();
  const il::Array<int>& asArrayOfInt32() const;
  il::Array<int>& asArrayOfInt32();
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

inline Dynamic::Dynamic() { type_ = il::Type::TNull; }

inline Dynamic::Dynamic(bool value) {
  type_ = il::Type::TBool;
  bool_val_ = value;
}

inline Dynamic::Dynamic(int value) {
  type_ = il::Type::TInteger;
  integer_val_ = value;
}

#ifdef IL_64_BIT
inline Dynamic::Dynamic(il::int_t value) {
  type_ = il::Type::TInteger;
  integer_val_ = value;
}
#endif

inline Dynamic::Dynamic(float value) {
  type_ = il::Type::TFloat;
  float_val_ = value;
}

inline Dynamic::Dynamic(double value) {
  type_ = il::Type::TDouble;
  double_val_ = value;
}

template <il::int_t m>
inline Dynamic::Dynamic(const char (&value)[m]) {
  type_ = il::Type::TString;
  string_val_ = new il::String{value};
}

inline Dynamic::Dynamic(il::StringType t, const char* value, il::int_t n) {
  type_ = il::Type::TString;
  string_val_ = new il::String{t, value, n};
}

inline Dynamic::Dynamic(const il::String& value) {
  type_ = il::Type::TString;
  string_val_ = new il::String{value};
}

inline Dynamic::Dynamic(il::String&& value) {
  type_ = il::Type::TString;
  string_val_ = new il::String{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<il::Dynamic>& value) {
  type_ = il::Type::TArray;
  array_val_ = new il::Array<il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Array<il::Dynamic>&& value) {
  type_ = il::Type::TArray;
  array_val_ = new il::Array<il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<unsigned char>& value) {
  type_ = il::Type::TArrayOfUint8;
  array_of_uint8_val_ = new il::Array<unsigned char>{value};
}

inline Dynamic::Dynamic(il::Array<unsigned char>&& value) {
  type_ = il::Type::TArrayOfUint8;
  array_of_uint8_val_ = new il::Array<unsigned char>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<int>& value) {
  type_ = il::Type::TArrayOfInt32;
  array_of_int32_val_ = new il::Array<int>{value};
}

inline Dynamic::Dynamic(il::Array<int>&& value) {
  type_ = il::Type::TArrayOfInt32;
  array_of_int32_val_ = new il::Array<int>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array<double>& value) {
  type_ = il::Type::TArrayOfDouble;
  array_of_double_val_ = new il::Array<double>{value};
}

inline Dynamic::Dynamic(il::Array<double>&& value) {
  type_ = il::Type::TArrayOfDouble;
  array_of_double_val_ = new il::Array<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<double>& value) {
  type_ = il::Type::TArray2dOfDouble;
  array2d_of_double_val_ = new il::Array2D<double>{value};
}

inline Dynamic::Dynamic(il::Array2D<double>&& value) {
  type_ = il::Type::TArray2dOfDouble;
  array2d_of_double_val_ = new il::Array2D<double>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Array2D<unsigned char>& value) {
  type_ = il::Type::TArray2dOfUint8;
  array2d_of_uint8_val_ = new il::Array2D<unsigned char>{value};
}

inline Dynamic::Dynamic(il::Array2D<unsigned char>&& value) {
  type_ = il::Type::TArray2dOfUint8;
  array2d_of_uint8_val_ = new il::Array2D<unsigned char>{std::move(value)};
}

inline Dynamic::Dynamic(const il::Map<il::String, il::Dynamic>& value) {
  type_ = il::Type::TMap;
  map_val_ = new il::Map<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::Map<il::String, il::Dynamic>&& value) {
  type_ = il::Type::TMap;
  map_val_ = new il::Map<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(const il::MapArray<il::String, il::Dynamic>& value) {
  type_ = il::Type::TMapArray;
  map_array_val_ = new il::MapArray<il::String, il::Dynamic>{value};
}

inline Dynamic::Dynamic(il::MapArray<il::String, il::Dynamic>&& value) {
  type_ = il::Type::TMapArray;
  map_array_val_ = new il::MapArray<il::String, il::Dynamic>{std::move(value)};
}

inline Dynamic::Dynamic(il::Type value) {
  type_ = value;
  switch (value) {
    case il::Type::TArrayOfDouble:
      array_of_double_val_ = new il::Array<double>{};
      break;
    case il::Type::TArrayOfInt32:
      array_of_int32_val_ = new il::Array<int>{};
      break;
    case il::Type::TMap:
      map_val_ = new il::Map<il::String, il::Dynamic>{};
      break;
    case il::Type::TMapArray:
      map_array_val_ = new il::MapArray<il::String, il::Dynamic>{};
      break;
    default:
      IL_UNREACHABLE;
  }
}

inline Dynamic::~Dynamic() { releaseMemory(); }

inline Dynamic::Dynamic(const il::Dynamic& other) {
  type_ = other.type_;
  if (type_ == il::Type::TString) {
    string_val_ = new il::String{*other.string_val_};
  } else if (type_ == il::Type::TArray) {
    array_val_ = new il::Array<il::Dynamic>{*other.array_val_};
  } else if (type_ == il::Type::TArrayOfUint8) {
    array_of_uint8_val_ =
        new il::Array<unsigned char>{*other.array_of_uint8_val_};
  } else if (type_ == il::Type::TArrayOfInt32) {
    array_of_int32_val_ = new il::Array<int>{*other.array_of_int32_val_};
  } else if (type_ == il::Type::TArrayOfDouble) {
    array_of_double_val_ = new il::Array<double>{*other.array_of_double_val_};
  } else if (type_ == il::Type::TArray2dOfDouble) {
    array2d_of_double_val_ =
        new il::Array2D<double>{*other.array2d_of_double_val_};
  } else if (type_ == il::Type::TMap) {
    map_val_ = new il::Map<il::String, il::Dynamic>{*other.map_val_};
  } else if (type_ == il::Type::TMapArray) {
    map_array_val_ =
        new il::MapArray<il::String, il::Dynamic>{*other.map_array_val_};
  } else {
    data_ = other.data_;
  }
}

inline Dynamic::Dynamic(il::Dynamic&& other) {
  type_ = other.type_;
  data_ = other.data_;
  other.type_ = il::Type::TNull;
}

inline il::Dynamic& Dynamic::operator=(const il::Dynamic& other) {
  releaseMemory();
  type_ = other.type_;
  if (type_ == il::Type::TString) {
    string_val_ = new il::String{*other.string_val_};
  } else if (type_ == il::Type::TArray) {
    array_val_ = new il::Array<il::Dynamic>{*other.array_val_};
  } else if (type_ == il::Type::TArrayOfUint8) {
    array_of_uint8_val_ =
        new il::Array<unsigned char>{*other.array_of_uint8_val_};
  } else if (type_ == il::Type::TArrayOfInt32) {
    array_of_int32_val_ = new il::Array<int>{*other.array_of_int32_val_};
  } else if (type_ == il::Type::TArrayOfDouble) {
    array_of_double_val_ = new il::Array<double>{*other.array_of_double_val_};
  } else if (type_ == il::Type::TArray2dOfDouble) {
    array2d_of_double_val_ =
        new il::Array2D<double>{*other.array2d_of_double_val_};
  } else if (type_ == il::Type::TMap) {
    map_val_ = new il::Map<il::String, il::Dynamic>{*other.map_val_};
  } else if (type_ == il::Type::TMapArray) {
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
  other.type_ = il::Type::TNull;
  return *this;
}

inline il::Type Dynamic::type() const { return type_; }

inline bool Dynamic::isNull() const { return type_ == il::Type::TNull; }

inline bool Dynamic::isBool() const { return type_ == il::Type::TBool; }

inline bool Dynamic::isInteger() const { return type_ == il::Type::TInteger; }

inline bool Dynamic::isFloat() const { return type_ == il::Type::TFloat; }

inline bool Dynamic::isDouble() const { return type_ == il::Type::TDouble; }

inline bool Dynamic::isString() const { return type_ == il::Type::TString; }

inline bool Dynamic::isArray() const { return type_ == il::Type::TArray; }

inline bool Dynamic::isArrayOfUint8() const {
  return type_ == il::Type::TArrayOfUint8;
}

inline bool Dynamic::isArrayOfInt32() const {
  return type_ == il::Type::TArrayOfInt32;
}

inline bool Dynamic::isArrayOfDouble() const {
  return type_ == il::Type::TArrayOfDouble;
}

inline bool Dynamic::isArray2dOfDouble() const {
  return type_ == il::Type::TArray2dOfDouble;
}

inline bool Dynamic::isArray2dOfUint8() const {
  return type_ == il::Type::TArray2dOfUint8;
}

inline bool Dynamic::isMap() const { return type_ == il::Type::TMap; }

inline bool Dynamic::isMapArray() const { return type_ == il::Type::TMapArray; }

inline bool Dynamic::toBool() const { return bool_val_; }

inline il::int_t Dynamic::toInteger() const { return integer_val_; }

inline float Dynamic::toFloat() const { return float_val_; }

inline double Dynamic::toDouble() const { return double_val_; }

inline const il::String& Dynamic::asString() const { return *string_val_; }

inline il::String& Dynamic::asString() { return *string_val_; }

inline const il::Array<il::Dynamic>& Dynamic::asArray() const {
  return *array_val_;
}

inline il::Array<il::Dynamic>& Dynamic::asArray() { return *array_val_; }

inline const il::Array<unsigned char>& Dynamic::asArrayOfUint8() const {
  return *array_of_uint8_val_;
}

inline il::Array<unsigned char>& Dynamic::asArrayOfUint8() {
  return *array_of_uint8_val_;
}

inline const il::Array<int>& Dynamic::asArrayOfInt32() const {
  return *array_of_int32_val_;
}

inline il::Array<int>& Dynamic::asArrayOfInt32() {
  return *array_of_int32_val_;
}

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
  if (type_ == il::Type::TString) {
    delete string_val_;
  } else if (type_ == il::Type::TArray) {
    delete array_val_;
  } else if (type_ == il::Type::TArrayOfUint8) {
    delete array_of_uint8_val_;
  } else if (type_ == il::Type::TArrayOfInt32) {
    delete array_of_int32_val_;
  } else if (type_ == il::Type::TArrayOfDouble) {
    delete array_of_double_val_;
  } else if (type_ == il::Type::TArray2dOfDouble) {
    delete array2d_of_double_val_;
  } else if (type_ == il::Type::TArray2dOfUint8) {
    delete array2d_of_uint8_val_;
  } else if (type_ == il::Type::TMap) {
    delete map_val_;
  } else if (type_ == il::Type::TMapArray) {
    delete map_array_val_;
  }
}

}  // namespace il

#endif  // IL_DYNAMIC_H
