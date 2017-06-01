//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_TYPE_H
#define IL_TYPE_H

namespace il {

enum class Type {
  bool_t,
  int8_t,
  uint8_t,
  int16_t,
  uint16_t,
  int32_t,
  uint32_t,
  int64_t,
  uint64_t,
  float16_t,
  float32_t,
  float64_t,
  ascii_string_t,
  utf8_string_t,
  utf16_string_t,
  utf32_string_t
};

}

#endif  // IL_TYPE_H
