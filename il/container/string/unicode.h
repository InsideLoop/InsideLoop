//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_UNICODE_H
#define IL_UNICODE_H

#include <il/container/string/String.h>
#include <il/container/string/U16String.h>

namespace il {

il::U16String to_u16(const il::String& string) {
  il::U16String ans{};
  
  for (il::int_t i = 0; i < string.size(); i = string.next_cp(i)) {
    ans.append(string.cp(i));
  }
  ans.append('\0');
  return ans;
};

enum Unicode : std::int32_t {
  snowman = 0x2603,
  mahjong_tile_red_dragon = 0x0001F004,
  grinning_face = 0x0001F600,
  grinning_face_with_smiling_eyes = 0x0001F601,
  smiling_face_with_horns = 0x0001F608
};

}

#endif  // IL_UNICODE_H
