//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_VIEW_STRING_H
#define IL_VIEW_STRING_H

#include <il/String.h>
#include <il/StringView.h>

namespace il {

inline ConstStringView view(const il::String& string) {
  return ConstStringView{string.as_c_string(), string.size()};
}

inline ConstStringView view(const char* string) {
  il::int_t size = 0;
  while (string[size] != '\0') {
    ++size;
  }

  return ConstStringView{string, size};
}

}  // namespace il

#endif  // IL_VIEW_STRING_H
