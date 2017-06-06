//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_PROTOBUF_H
#define IL_PROTOBUF_H

#include <cstdint>

#include <il/SmallArray.h>

namespace il {

enum class WireType : std::uint8_t {
  varint = 0,
  length_delimited = 2,
  size_32 = 5,
  size_64 = 1
};

il::SmallArray<std::uint8_t, 10> compress_varint(std::uint64_t n) {
  const std::uint64_t max_byte = 1 << 7;
  const std::uint8_t continuation_byte = 0x80;
  il::SmallArray<std::uint8_t, 10> ans{};
  while (true) {
    std::uint8_t r = static_cast<std::uint8_t>(n % max_byte);
    n /= max_byte;
    if (n == 0) {
      ans.append(r);
      break;
    } else {
      r |= continuation_byte;
      ans.append(r);
    }
  }
  return ans;
}

std::uint64_t decompress_varint(const il::SmallArray<std::uint8_t, 10>& v) {
  std::uint64_t n = 0;
  std::uint64_t multiplier = 1;
  const std::uint64_t max_byte = 1 << 7;
  const std::uint8_t continuation_mask = 0x7F;
  for (il::int_t i = 0; i < v.size(); ++i) {
    n += multiplier * (v[i] & continuation_mask);
    multiplier *= max_byte;
  }
  return n;
}

//void get_field_number_and_wire_type(const il::SmallArray<std::uint8_t, 10>& v,
//                                    il::io_t, il::int_t& field_number,
//                                    il::WireType& wire_type) {
//  IL_EXPECT_MEDIUM(v.size() > 0);
//
//  const std::uint64_t n = decompress_varint(v);
//  const std::uint64_t wire_type_mask = 0x07;
//  wire_type = static_cast<std::uint8_t>(n & wire_type_mask);
//  field_number = static_cast<il::int_t>(n / 8);
//}

}

#endif  // IL_PROTOBUF_H
