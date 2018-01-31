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
  TVoid = 12,
  TBool = 0,
  TUInt8 = 1,
  TInt8 = 2,
  TUChar = 1,  // C type (unsigned char)
  TSChar = 2,  // C type (signed char, not char)
  TUInt16 = 3,
  TInt16 = 4,
  TUShort = 3,  // C type (unsigned short)
  TShort = 4,   // C type (shor)
  TUInt32 = 5,
  TInt32 = 6,
  TUInt = 5,  // C type (unsigned int)
  TInt = 6,   // C type (int)
  TUInt64 = 7,
  TInt64 = 8,
#ifdef IL_64_BIT
  TUInteger = 7,  // C++ type (il::uint_t aka std::size_t)
  TInteger = 8,   // C++ type (il::int_t aka std:ptrdiff_t)
#else
  TUInteger = 5,  // C++ type (il::uint_t aka std::size_t)
  TInteger = 6,   // C++ type (il::int_t aka std:ptrdiff_t)
#endif
  // TFp16 = 9,
  TFp32 = 10,
  TFp64 = 11,
  TFloat = 10,   // C type (float)
  TDouble = 11,  // C type (double)
  TFloatingPoint = 11,

  TString = 13,
  TArray = 14,
  TMap = 15,
  TMapArray = 16,

  TArrayOfBool = 20,
  TArrayOfUInt8 = 21,
  TArrayOfInt8 = 22,
  TArrayOfUInt16 = 23,
  TArrayOfInt16 = 24,
  TArrayOfUInt32 = 25,
  TArrayOfInt32 = 26,
  TArrayOfUInt = 25,
  TArrayOfInt = 26,
  TArrayOfUInt64 = 27,
  TArrayOfInt64 = 28,
#ifdef IL_64_BIT
  TArrayOfUInteger = 27,
  TArrayOfInteger = 28,
#else
  TArrayOfUInteger = 25,
  TArrayOfInteger = 26,
#endif
  TArrayOfFp32 = 30,
  TArrayOfFp64 = 31,
  TArrayOfFloat = 30,
  TArrayOfDouble = 31,
  TArrayOfFloatingPoint = 31,
  TArrayOfString = 32,
  TArrayOfStruct = 33,

  TArrayViewOfBool = 20,
  TArrayViewOfUInt8 = 21,
  TArrayViewOfInt8 = 22,
  TArrayViewOfUInt16 = 23,
  TArrayViewOfInt16 = 24,
  TArrayViewOfUInt32 = 25,
  TArrayViewOfInt32 = 26,
  TArrayViewOfUInt = 25,
  TArrayViewOfInt = 26,
  TArrayViewOfUInt64 = 27,
  TArrayViewOfInt64 = 28,
#ifdef IL_64_BIT
  TArrayViewOfUInteger = 27,
  TArrayViewOfInteger = 28,
#else
  TArrayViewOfUInteger = 25,
  TArrayViewOfInteger = 26,
#endif
  TArrayViewOfFp32 = 30,
  TArrayViewOfFp64 = 31,
  TArrayViewOfFloat = 30,
  TArrayViewOfDouble = 31,
  TArrayViewOfFloatingPoint = 31,
  TArrayViewOfString = 32,

  TArrayEditOfBool = 20,
  TArrayEditOfUInt8 = 21,
  TArrayEditOfInt8 = 22,
  TArrayEditOfUInt16 = 23,
  TArrayEditOfInt16 = 24,
  TArrayEditOfUInt32 = 25,
  TArrayEditOfInt32 = 26,
  TArrayEditOfUInt = 25,
  TArrayEditOfInt = 26,
  TArrayEditOfUInt64 = 27,
  TArrayEditOfInt64 = 28,
#ifdef IL_64_BIT
  TArrayEditOfUInteger = 27,
  TArrayEditOfInteger = 28,
#else
  TArrayEditOfUInteger = 25,
  TArrayEditOfInteger = 26,
#endif
  TArrayEditOfFp32 = 30,
  TArrayEditOfFp64 = 31,
  TArrayEditOfFloat = 30,
  TArrayEditOfDouble = 31,
  TArrayEditOfFloatingPoint = 31,
  TArrayEditOfString = 32,

  TArray2DOfBool = 40,
  TArray2DOfUInt8 = 41,
  TArray2DOfInt8 = 42,
  TArray2DOfUInt16 = 43,
  TArray2DOfInt16 = 44,
  TArray2DOfUInt32 = 45,
  TArray2DOfInt32 = 46,
  TArray2DOfUInt = 45,
  TArray2DOfInt = 46,
  TArray2DOfUInt64 = 47,
  TArray2DOfInt64 = 48,
#ifdef IL_64_BIT
  TArray2DOfUInteger = 47,
  TArray2DOfInteger = 48,
#else
  TArray2DOfUInteger = 45,
  TArray2DOfInteger = 46,
#endif
  // TArray2DOfFp16 = 49,
  TArray2DOfFp32 = 50,
  TArray2DOfFp64 = 51,
  TArray2DOfFloat = 50,
  TArray2DOfDouble = 51,
  TArray2DOfFloatingPoint = 51,
  TArray2DOfString = 52,

  TArray2COfBool = 60,
  TArray2COfUInt8 = 61,
  TArray2COfInt8 = 62,
  TArray2COfUInt16 = 63,
  TArray2COfInt16 = 64,
  TArray2COfUInt32 = 65,
  TArray2COfInt32 = 66,
  TArray2COfUInt = 65,
  TArray2COfInt = 66,
  TArray2COfUInt64 = 67,
  TArray2COfInt64 = 68,
#ifdef IL_64_BIT
  TArray2COfUInteger = 67,
  TArray2COfInteger = 68,
#else
  TArray2COfUInteger = 65,
  TArray2COfInteger = 66,
#endif
  // TArray2COfFp16 = 69,
  TArray2COfFp32 = 70,
  TArray2COfFp64 = 71,
  TArray2COfFloat = 70,
  TArray2COfDouble = 71,
  TArray2COfFloatingPoint = 71,
  TArray2COfString = 72,

};

class Dynamic;

template <typename T>
il::Type typeId() {
  return il::Type::TVoid;
}

template <>
inline il::Type typeId<bool>() {
  return il::Type::TBool;
}

template <>
inline il::Type typeId<int>() {
  return il::Type::TInt;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::int_t>() {
  return il::Type::TInteger;
}
#endif

template <>
inline il::Type typeId<float>() {
  return il::Type::TFloat;
}

template <>
inline il::Type typeId<double>() {
  return il::Type::TDouble;
}

template <>
inline il::Type typeId<il::String>() {
  return il::Type::TString;
}

template <>
inline il::Type typeId<il::Array<il::Dynamic>>() {
  return il::Type::TArray;
}

template <>
inline il::Type typeId<il::Map<il::String, il::Dynamic>>() {
  return il::Type::TMap;
}

template <>
inline il::Type typeId<il::MapArray<il::String, il::Dynamic>>() {
  return il::Type::TMapArray;
}

template <>
inline il::Type typeId<il::Array<unsigned char>>() {
  return il::Type::TArrayOfUInt8;
}

template <>
inline il::Type typeId<il::Array<signed char>>() {
  return il::Type::TArrayOfInt8;
}

template <>
inline il::Type typeId<il::Array<unsigned short>>() {
  return il::Type::TArrayOfUInt16;
}

template <>
inline il::Type typeId<il::Array<short>>() {
  return il::Type::TArrayOfInt16;
}

template <>
inline il::Type typeId<il::Array<unsigned int>>() {
  return il::Type::TArrayOfUInt32;
}

template <>
inline il::Type typeId<il::Array<int>>() {
  return il::Type::TArrayOfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array<std::size_t>>() {
  return il::Type::TArrayOfUInt64;
}

template <>
inline il::Type typeId<il::Array<il::int_t>>() {
  return il::Type::TArrayOfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array<float>>() {
  return il::Type::TArrayOfFloat;
}

template <>
inline il::Type typeId<il::Array<double>>() {
  return il::Type::TArrayOfDouble;
}

template <>
inline il::Type typeId<il::Array2D<unsigned char>>() {
  return il::Type::TArray2DOfUInt8;
}

template <>
inline il::Type typeId<il::Array2D<signed char>>() {
  return il::Type::TArray2DOfInt8;
}

template <>
inline il::Type typeId<il::Array2D<unsigned short>>() {
  return il::Type::TArray2DOfUInt16;
}

template <>
inline il::Type typeId<il::Array2D<short>>() {
  return il::Type::TArray2DOfInt16;
}

template <>
inline il::Type typeId<il::Array2D<unsigned int>>() {
  return il::Type::TArray2DOfUInt32;
}

template <>
inline il::Type typeId<il::Array2D<int>>() {
  return il::Type::TArray2DOfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array2D<std::size_t>>() {
  return il::Type::TArray2DOfUInt64;
}

template <>
inline il::Type typeId<il::Array2D<il::int_t>>() {
  return il::Type::TArray2DOfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array2D<float>>() {
  return il::Type::TArray2DOfFloat;
}

template <>
inline il::Type typeId<il::Array2D<double>>() {
  return il::Type::TArray2DOfDouble;
}

template <>
inline il::Type typeId<il::Array2C<signed char>>() {
  return il::Type::TArray2COfInt8;
}

template <>
inline il::Type typeId<il::Array2C<unsigned short>>() {
  return il::Type::TArray2COfUInt16;
}

template <>
inline il::Type typeId<il::Array2C<short>>() {
  return il::Type::TArray2COfInt16;
}

template <>
inline il::Type typeId<il::Array2C<unsigned int>>() {
  return il::Type::TArray2COfUInt32;
}

template <>
inline il::Type typeId<il::Array2C<int>>() {
  return il::Type::TArray2COfInt32;
}

#ifdef IL_64_BIT
template <>
inline il::Type typeId<il::Array2C<std::size_t>>() {
  return il::Type::TArray2COfUInt64;
}

template <>
inline il::Type typeId<il::Array2C<il::int_t>>() {
  return il::Type::TArray2COfInt64;
}
#endif

template <>
inline il::Type typeId<il::Array2C<float>>() {
  return il::Type::TArray2COfFloat;
}

template <>
inline il::Type typeId<il::Array2C<double>>() {
  return il::Type::TArray2COfDouble;
}

}  // namespace il

#endif  // IL_TYPE_H
