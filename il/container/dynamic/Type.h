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

#include <il/String.h>
#include <il/Array.h>
#include <il/Array2D.h>
#include <il/Array2C.h>
#include <il/Map.h>
#include <il/MapArray.h>

namespace il {

enum class Type : unsigned char {
  TBool = 0,
  TUint8 = 1,
  TInt8 = 2,
  TUchar = 1,   // C type (unsigned char)
  TSchar = 2,   // C type (signed char, not char)
  TUint16 = 3,
  TInt16 = 4,
  TUShort = 3,  // C type (unsigned short)
  TShort = 4,   // C type (shor)
  TUint32 = 5,
  TInt32 = 6,
  TUint = 5,    // C type (unsigned int)
  TInt = 6,     // C type (int)
  TUint64 = 7,
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
  TFloat = 10,    // C type (float)
  TDouble = 11,   // C type (double)
  TFloatingPoint = 11,
  TString = 12,
  TArray = 13,
  TMap = 14,
  TMapArray = 15,
  TArrayOfBool = 20,
  TArrayOfUint8 = 21,
  TArrayOfInt8 = 22,
  TArrayOfUint16 = 23,
  TArrayOfInt16 = 24,
  TArrayOfUint32 = 25,
  TArrayOfInt32 = 26,
  TArrayOfUint = 25,
  TArrayOfInt = 26,
  TArrayOfUint64 = 27,
  TArrayOfInt64 = 28,
#ifdef IL_64_BIT
  TArrayOfUInteger = 27,
  TArrayOfInteger = 28,
#else
  TArrayOfUInteger = 25,
  TArrayOfInteger = 26,
#endif
  // TArrayOfFp16 = 29,
      TArrayOfFp32 = 30,
  TArrayOfFp64 = 31,
  TArrayOfFloat = 30,
  TArrayOfDouble = 31,
  TArrayOfFloatingPoint = 31,
  TArrayOfString = 32,

  TArray2dOfBool = 40,
  TArray2dOfUint8 = 41,
  TArray2dOfInt8 = 42,
  TArray2dOfUint16 = 43,
  TArray2dOfInt16 = 44,
  TArray2dOfUint32 = 45,
  TArray2dOfInt32 = 46,
  TArray2dOfUint = 45,
  TArray2dOfInt = 46,
  TArray2dOfUint64 = 47,
  TArray2dOfInt64 = 48,
#ifdef IL_64_BIT
  TArray2dOfUInteger = 47,
  TArray2dOfInteger = 48,
#else
  TArray2dOfUInteger = 45,
  TArray2dOfInteger = 46,
#endif
  // TArray2dOfFp16 = 49,
      TArray2dOfFp32 = 50,
  TArray2dOfFp64 = 51,
  TArray2dOfFloat = 50,
  TArray2dOfDouble = 51,
  TArray2dOfFloatingPoint = 51,
  TArray2dOfString = 52,

  TArray2cOfBool = 60,
  TArray2cOfUint8 = 61,
  TArray2cOfInt8 = 62,
  TArray2cOfUint16 = 63,
  TArray2cOfInt16 = 64,
  TArray2cOfUint32 = 65,
  TArray2cOfInt32 = 66,
  TArray2cOfUint = 65,
  TArray2cOfInt = 66,
  TArray2cOfUint64 = 67,
  TArray2cOfInt64 = 68,
#ifdef IL_64_BIT
  TArray2cOfUInteger = 67,
  TArray2cOfInteger = 68,
#else
  TArray2cOfUInteger = 65,
  TArray2cOfInteger = 66,
#endif
  // TArray2cOfFp16 = 69,
      TArray2cOfFp32 = 70,
  TArray2cOfFp64 = 71,
  TArray2cOfFloat = 70,
  TArray2cOfDouble = 71,
  TArray2cOfFloatingPoint = 71,
  TArray2cOfString = 72,

  TUnknown = 255
};

template<typename T>
il::Type typeId() {
  return il::Type::TUnknown;
}

template<>
inline il::Type typeId<bool>() {
  return il::Type::TBool;
}

template<>
inline il::Type typeId<float>() {
  return il::Type::TFloat;
}

template<>
inline il::Type typeId<double>() {
  return il::Type::TDouble;
}

template<>
inline il::Type typeId<il::String>() {
  return il::Type::TString;
}

template<>
inline il::Type typeId<il::Array<unsigned char>>() {
  return il::Type::TArrayOfUint8;
}

template<>
inline il::Type typeId<il::Array<signed char>>() {
  return il::Type::TArrayOfInt8;
}

template<>
inline il::Type typeId<il::Array<unsigned short>>() {
  return il::Type::TArrayOfUint16;
}

template<>
inline il::Type typeId<il::Array<short>>() {
  return il::Type::TArrayOfInt16;
}

template<>
inline il::Type typeId<il::Array<unsigned int>>() {
  return il::Type::TArrayOfUint32;
}

template<>
inline il::Type typeId<il::Array<int>>() {
  return il::Type::TArrayOfInt32;
}

template<>
inline il::Type typeId<il::Array<std::size_t>>() {
  return il::Type::TArrayOfUint64;
}

template<>
inline il::Type typeId<il::Array<il::int_t>>() {
  return il::Type::TArrayOfInt64;
}

template<>
inline il::Type typeId<il::Array<float>>() {
  return il::Type::TArrayOfFloat;
}

template<>
inline il::Type typeId<il::Array<double>>() {
  return il::Type::TArrayOfDouble;
}

template<>
inline il::Type typeId<il::Array2D<unsigned char>>() {
  return il::Type::TArray2dOfUint8;
}

template<>
inline il::Type typeId<il::Array2D<signed char>>() {
  return il::Type::TArray2dOfInt8;
}

template<>
inline il::Type typeId<il::Array2D<unsigned short>>() {
  return il::Type::TArray2dOfUint16;
}

template<>
inline il::Type typeId<il::Array2D<short>>() {
  return il::Type::TArray2dOfInt16;
}

template<>
inline il::Type typeId<il::Array2D<unsigned int>>() {
  return il::Type::TArray2dOfUint32;
}

template<>
inline il::Type typeId<il::Array2D<int>>() {
  return il::Type::TArray2dOfInt32;
}

template<>
inline il::Type typeId<il::Array2D<std::size_t>>() {
  return il::Type::TArray2dOfUint64;
}

template<>
inline il::Type typeId<il::Array2D<il::int_t>>() {
  return il::Type::TArray2dOfInt64;
}

template<>
inline il::Type typeId<il::Array2D<float>>() {
  return il::Type::TArray2dOfFloat;
}

template<>
inline il::Type typeId<il::Array2D<double>>() {
  return il::Type::TArray2dOfDouble;
}

template<>
inline il::Type typeId<il::Array2C<signed char>>() {
  return il::Type::TArray2cOfInt8;
}

template<>
inline il::Type typeId<il::Array2C<unsigned short>>() {
  return il::Type::TArray2cOfUint16;
}

template<>
inline il::Type typeId<il::Array2C<short>>() {
  return il::Type::TArray2cOfInt16;
}

template<>
inline il::Type typeId<il::Array2C<unsigned int>>() {
  return il::Type::TArray2cOfUint32;
}

template<>
inline il::Type typeId<il::Array2C<int>>() {
  return il::Type::TArray2cOfInt32;
}

template<>
inline il::Type typeId<il::Array2C<std::size_t>>() {
  return il::Type::TArray2cOfUint64;
}

template<>
inline il::Type typeId<il::Array2C<il::int_t>>() {
  return il::Type::TArray2cOfInt64;
}

template<>
inline il::Type typeId<il::Array2C<float>>() {
  return il::Type::TArray2cOfFloat;
}

template<>
inline il::Type typeId<il::Array2C<double>>() {
  return il::Type::TArray2cOfDouble;
}

}

#endif // IL_TYPE_H
