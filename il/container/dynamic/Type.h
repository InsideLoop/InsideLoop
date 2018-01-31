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

#ifndef IL_TYPE_H
#define IL_TYPE_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/Array2D.h>
#include <il/Map.h>
#include <il/MapArray.h>
#include <il/String.h>

namespace il {

enum class Type : unsigned char {
  kVoid = 12,
  kBool = 0,
  kUInt8 = 1,
  kInt8 = 2,
  kUChar = 1,  // C type (unsigned char)
  kSChar = 2,  // C type (signed char, not char)
  kUInt16 = 3,
  kInt16 = 4,
  kUShort = 3,  // C type (unsigned short)
  kShort = 4,   // C type (shor)
  kUInt32 = 5,
  kInt32 = 6,
  kUInt = 5,  // C type (unsigned int)
  kInt = 6,   // C type (int)
  kUInt64 = 7,
  kInt64 = 8,
#ifdef IL_64_BIT
  kUInteger = 7,  // C++ type (il::uint_t aka std::size_t)
  kInteger = 8,   // C++ type (il::int_t aka std:ptrdiff_t)
#else
  kUInteger = 5,  // C++ type (il::uint_t aka std::size_t)
  kInteger = 6,   // C++ type (il::int_t aka std:ptrdiff_t)
#endif
  // TFp16 = 9,
  kFp32 = 10,
  kFp64 = 11,
  kFloat = 10,   // C type (float)
  kDouble = 11,  // C type (double)
  kFloatingPoint = 11,

  kString = 13,
  kArray = 14,
  kMap = 15,
  kMapArray = 16,

  kArrayOfBool = 20,
  kArrayOfUInt8 = 21,
  kArrayOfInt8 = 22,
  kArrayOfUInt16 = 23,
  kArrayOfInt16 = 24,
  kArrayOfUInt32 = 25,
  kArrayOfInt32 = 26,
  kArrayOfUInt = 25,
  kArrayOfInt = 26,
  kArrayOfUInt64 = 27,
  kArrayOfInt64 = 28,
#ifdef IL_64_BIT
  kArrayOfUInteger = 27,
  kArrayOfInteger = 28,
#else
  kArrayOfUInteger = 25,
  kArrayOfInteger = 26,
#endif
  kArrayOfFp32 = 30,
  kArrayOfFp64 = 31,
  kArrayOfFloat = 30,
  kArrayOfDouble = 31,
  kArrayOfFloatingPoint = 31,
  kArrayOfString = 32,
  kArrayOfStruct = 33,

  kArrayViewOfBool = 20,
  kArrayViewOfUInt8 = 21,
  kArrayViewOfInt8 = 22,
  kArrayViewOfUInt16 = 23,
  kArrayViewOfInt16 = 24,
  kArrayViewOfUInt32 = 25,
  kArrayViewOfInt32 = 26,
  kArrayViewOfUInt = 25,
  kArrayViewOfInt = 26,
  kArrayViewOfUInt64 = 27,
  kArrayViewOfInt64 = 28,
#ifdef IL_64_BIT
  kArrayViewOfUInteger = 27,
  kArrayViewOfInteger = 28,
#else
  kArrayViewOfUInteger = 25,
  kArrayViewOfInteger = 26,
#endif
  kArrayViewOfFp32 = 30,
  kArrayViewOfFp64 = 31,
  kArrayViewOfFloat = 30,
  kArrayViewOfDouble = 31,
  kArrayViewOfFloatingPoint = 31,
  kArrayViewOfString = 32,

  kArrayEditOfBool = 20,
  kArrayEditOfUInt8 = 21,
  kArrayEditOfInt8 = 22,
  kArrayEditOfUInt16 = 23,
  kArrayEditOfInt16 = 24,
  kArrayEditOfUInt32 = 25,
  kArrayEditOfInt32 = 26,
  kArrayEditOfUInt = 25,
  kArrayEditOfInt = 26,
  kArrayEditOfUInt64 = 27,
  kArrayEditOfInt64 = 28,
#ifdef IL_64_BIT
  kArrayEditOfUInteger = 27,
  kArrayEditOfInteger = 28,
#else
  kArrayEditOfUInteger = 25,
  kArrayEditOfInteger = 26,
#endif
  kArrayEditOfFp32 = 30,
  kArrayEditOfFp64 = 31,
  kArrayEditOfFloat = 30,
  kArrayEditOfDouble = 31,
  kArrayEditOfFloatingPoint = 31,
  kArrayEditOfString = 32,

  kArray2DOfBool = 40,
  kArray2DOfUInt8 = 41,
  kArray2DOfInt8 = 42,
  kArray2DOfUInt16 = 43,
  kArray2DOfInt16 = 44,
  kArray2DOfUInt32 = 45,
  kArray2DOfInt32 = 46,
  kArray2DOfUInt = 45,
  kArray2DOfInt = 46,
  kArray2DOfUInt64 = 47,
  kArray2DOfInt64 = 48,
#ifdef IL_64_BIT
  kArray2DOfUInteger = 47,
  kArray2DOfInteger = 48,
#else
  kArray2DOfUInteger = 45,
  kArray2DOfInteger = 46,
#endif
  // TArray2DOfFp16 = 49,
  kArray2DOfFp32 = 50,
  kArray2DOfFp64 = 51,
  kArray2DOfFloat = 50,
  kArray2DOfDouble = 51,
  kArray2DOfFloatingPoint = 51,
  kArray2DOfString = 52,

  kArray2COfBool = 60,
  kArray2COfUInt8 = 61,
  kArray2COfInt8 = 62,
  kArray2COfUInt16 = 63,
  kArray2COfInt16 = 64,
  kArray2COfUInt32 = 65,
  kArray2COfInt32 = 66,
  kArray2COfUInt = 65,
  kArray2COfInt = 66,
  kArray2COfUInt64 = 67,
  kArray2COfInt64 = 68,
#ifdef IL_64_BIT
  kArray2COfUInteger = 67,
  kArray2COfInteger = 68,
#else
  kArray2COfUInteger = 65,
  kArray2COfInteger = 66,
#endif
  // TArray2COfFp16 = 69,
  kArray2COfFp32 = 70,
  kArray2COfFp64 = 71,
  kArray2COfFloat = 70,
  kArray2COfDouble = 71,
  kArray2COfFloatingPoint = 71,
  kArray2COfString = 72,

};

class Dynamic;

template <typename T>
il::Type typeId() {
  return il::Type::kVoid;
}

template <>
inline il::Type typeId<bool>() {
  return il::Type::kBool;
}

template <>
inline il::Type typeId<int>() {
  return il::Type::kInt;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::int_t>() {
  return il::Type::kInteger;
}
#endif

template <>
inline il::Type typeId<float>() {
  return il::Type::kFloat;
}

template <>
inline il::Type typeId<double>() {
  return il::Type::kDouble;
}

template <>
inline il::Type typeId<il::String>() {
  return il::Type::kString;
}

template <>
inline il::Type typeId<il::Array<il::Dynamic>>() {
  return il::Type::kArray;
}

template <>
inline il::Type typeId<il::Map<il::String, il::Dynamic>>() {
  return il::Type::kMap;
}

template <>
inline il::Type typeId<il::MapArray<il::String, il::Dynamic>>() {
  return il::Type::kMapArray;
}

template <>
inline il::Type typeId<il::Array<unsigned char>>() {
  return il::Type::kArrayOfUInt8;
}

template <>
inline il::Type typeId<il::Array<signed char>>() {
  return il::Type::kArrayOfInt8;
}

template <>
inline il::Type typeId<il::Array<unsigned short>>() {
  return il::Type::kArrayOfUInt16;
}

template <>
inline il::Type typeId<il::Array<short>>() {
  return il::Type::kArrayOfInt16;
}

template <>
inline il::Type typeId<il::Array<unsigned int>>() {
  return il::Type::kArrayOfUInt32;
}

template <>
inline il::Type typeId<il::Array<int>>() {
  return il::Type::kArrayOfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array<std::size_t>>() {
  return il::Type::kArrayOfUInt64;
}

template <>
inline il::Type typeId<il::Array<il::int_t>>() {
  return il::Type::kArrayOfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array<float>>() {
  return il::Type::kArrayOfFloat;
}

template <>
inline il::Type typeId<il::Array<double>>() {
  return il::Type::kArrayOfDouble;
}

template <>
inline il::Type typeId<il::Array2D<unsigned char>>() {
  return il::Type::kArray2DOfUInt8;
}

template <>
inline il::Type typeId<il::Array2D<signed char>>() {
  return il::Type::kArray2DOfInt8;
}

template <>
inline il::Type typeId<il::Array2D<unsigned short>>() {
  return il::Type::kArray2DOfUInt16;
}

template <>
inline il::Type typeId<il::Array2D<short>>() {
  return il::Type::kArray2DOfInt16;
}

template <>
inline il::Type typeId<il::Array2D<unsigned int>>() {
  return il::Type::kArray2DOfUInt32;
}

template <>
inline il::Type typeId<il::Array2D<int>>() {
  return il::Type::kArray2DOfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array2D<std::size_t>>() {
  return il::Type::kArray2DOfUInt64;
}

template <>
inline il::Type typeId<il::Array2D<il::int_t>>() {
  return il::Type::kArray2DOfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array2D<float>>() {
  return il::Type::kArray2DOfFloat;
}

template <>
inline il::Type typeId<il::Array2D<double>>() {
  return il::Type::kArray2DOfDouble;
}

template <>
inline il::Type typeId<il::Array2C<signed char>>() {
  return il::Type::kArray2COfInt8;
}

template <>
inline il::Type typeId<il::Array2C<unsigned short>>() {
  return il::Type::kArray2COfUInt16;
}

template <>
inline il::Type typeId<il::Array2C<short>>() {
  return il::Type::kArray2COfInt16;
}

template <>
inline il::Type typeId<il::Array2C<unsigned int>>() {
  return il::Type::kArray2COfUInt32;
}

template <>
inline il::Type typeId<il::Array2C<int>>() {
  return il::Type::kArray2COfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array2C<std::size_t>>() {
  return il::Type::kArray2COfUInt64;
}

template <>
inline il::Type typeId<il::Array2C<il::int_t>>() {
  return il::Type::kArray2COfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array2C<float>>() {
  return il::Type::kArray2COfFloat;
}

template <>
inline il::Type typeId<il::Array2C<double>>() {
  return il::Type::kArray2COfDouble;
}

}  // namespace il

#endif  // IL_TYPE_H
