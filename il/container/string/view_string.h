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

namespace il {

inline StringView view(const il::String& string) {
  return StringView{string.type(), string.asCString(), string.size()};
}

inline StringView view(const char* string) {
  il::int_t size = 0;
  while (string[size] != '\0') {
    ++size;
  }

  return StringView{il::StringType::Bytes, string, size};
}

}  // namespace il

#endif  // IL_VIEW_STRING_H
