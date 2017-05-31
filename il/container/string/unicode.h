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

enum CodePoint : std::int32_t {
  snowman = 0x2603,
  mahjong_tile_red_dragon = 0x0001F004,
  grinning_face = 0x0001F600,
  grinning_face_with_smiling_eyes = 0x0001F601,
  smiling_face_with_horns = 0x0001F608,
  greek_small_letter_alpha = 0x03B1,
  greek_small_letter_beta = 0x03B2,
  greek_small_letter_gamma = 0x03B3,
  greek_small_letter_delta = 0x03B4,
  greek_small_letter_epsilon = 0x03B5,
  greek_small_letter_zeta = 0x03B6,
  greek_small_letter_eta = 0x03B7,
  greek_small_letter_theta = 0x03B8,
  greek_small_letter_iota = 0x03B9,
  greek_small_letter_kappa = 0x03BA,
  greek_small_letter_lambda = 0x03BB,
  greek_small_letter_mu = 0x03BC,
  greek_small_letter_nu = 0x03BD,
  greek_small_letter_xi = 0x03BE
};

}  // namespace il

#endif  // IL_UNICODE_H
