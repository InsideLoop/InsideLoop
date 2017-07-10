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
#include <il/container/string/StringView.h>
#include <il/container/string/UTF16String.h>

namespace il {

inline il::UTF16String toUtf16(const il::String& string) {
  il::UTF16String ans{};
  il::ConstStringView view{string.data(), string.size()};

  for (il::int_t i = 0; i < view.size(); i = view.nextRune(i)) {
    ans.append(view.toRune(i));
  }
  ans.append('\0');
  return ans;
}

enum Rune : int {
  Snowman = 0x2603,
  MahjongTileRedDragon = 0x0001F004,
  GrinningFace = 0x0001F600,
  GrinningFaceWithSmilingEyes = 0x0001F601,
  SmilingFaceWithHorns = 0x0001F608,
  ForAll = 0x2200,
  PartialDifferential = 0x2202,
  ThereExists = 0x2203,
  ThereDoesNotExist = 0x2204,
  EmptySet = 0x2205,
  Nabla = 0x2207,
  ElementOf = 0x2208,
  NotAnElementOf = 0x2209,
  RingOperator = 0x2218,
  Infinity = 0x221E,
  DoublestruckCapital_r = 0x211D,
  GreekSmallLetterAlpha = 0x03B1,
  GreekSmallLetterBeta = 0x03B2,
  GreekSmallLetterGamma = 0x03B3,
  GreekSmallLetterDelta = 0x03B4,
  GreekSmallLetterEpsilon = 0x03B5,
  GreekSmallLetterZeta = 0x03B6,
  GreekSmallLetterEta = 0x03B7,
  GreekSmallLetterTheta = 0x03B8,
  GreekSmallLetterIota = 0x03B9,
  GreekSmallLetterKappa = 0x03BA,
  reekSmallLetterLambda = 0x03BB,
  GreekSmallLetterMu = 0x03BC,
  GreekSmallLetterNu = 0x03BD,
  GreekSmallLetterXi = 0x03BE
};

}  // namespace il

#endif  // IL_UNICODE_H
