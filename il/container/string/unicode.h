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
  il::ConstStringView view{string.asCString(), string.size()};

  for (il::int_t i = 0; i < view.size(); i = view.nextCodePoint(i)) {
    ans.append(view.toCodePoint(i));
  }
  ans.append('\0');
  return ans;
}

enum CodePoint : int {
  kSnowman = 0x2603,
  kMahjongTileRedDragon = 0x0001F004,
  kGrinningFace = 0x0001F600,
  kGrinningFaceWithSmilingEyes = 0x0001F601,
  kSmilingFaceWithHorns = 0x0001F608,
  kForAll = 0x2200,
  kPartialDifferential = 0x2202,
  kThereExists = 0x2203,
  kThereDoesNotExist = 0x2204,
  kEmptySet = 0x2205,
  kNabla = 0x2207,
  kElementOf = 0x2208,
  kNotAnElementOf = 0x2209,
  kRingOperator = 0x2218,
  kInfinity = 0x221E,
  kDoublestruckCapital_r = 0x211D,
  kGreekSmallLetterAlpha = 0x03B1,
  kGreekSmallLetterBeta = 0x03B2,
  kGreekSmallLetterGamma = 0x03B3,
  kGreekSmallLetterDelta = 0x03B4,
  kGreekSmallLetterEpsilon = 0x03B5,
  kGreekSmallLetterZeta = 0x03B6,
  kGreekSmallLetterEta = 0x03B7,
  kGreekSmallLetterTheta = 0x03B8,
  kGreekSmallLetterIota = 0x03B9,
  kGreekSmallLetterKappa = 0x03BA,
  kGreekSmallLetterLambda = 0x03BB,
  kGreekSmallLetterMu = 0x03BC,
  kGreekSmallLetterNu = 0x03BD,
  kGreekSmallLetterXi = 0x03BE
};

}  // namespace il

#endif  // IL_UNICODE_H
