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

#ifndef IL_SPOT_T_H
#define IL_SPOT_T_H

#include <il/core.h>

namespace il {

class spot_t {
 private:
  il::int_t i_;
#ifdef IL_DEBUG_CLASS
  std::size_t hash_;
#endif

 public:
#ifdef IL_DEBUG_CLASS
  spot_t(il::int_t i, std::size_t hash);
#else
  spot_t(il::int_t i);
#endif
#ifdef IL_DEBUG_CLASS
  void SetIndex(il::int_t i, std::size_t hash);
#else
  void SetIndex(il::int_t i);
#endif
  il::int_t index() const;
  bool isValid() const;
#ifdef IL_DEBUG_CLASS
  std::size_t hash() const;
#endif
};

#ifdef IL_DEBUG_CLASS
inline spot_t::spot_t(il::int_t i, std::size_t hash) {
  i_ = i;
  hash_ = hash;
}
#else
inline spot_t::spot_t(il::int_t i) { i_ = i; }
#endif

#ifdef IL_DEBUG_CLASS
inline void spot_t::SetIndex(il::int_t i, std::size_t hash) {
  i_ = i;
  hash_ = hash;
}
#else
inline void spot_t::SetIndex(il::int_t i) { i_ = i; }
#endif

inline il::int_t spot_t::index() const { return i_; }

inline bool spot_t::isValid() const { return i_ >= 0; }

inline std::size_t spot_t::hash() const { return hash_; }

inline bool operator==(il::spot_t i0, il::spot_t i1) {
#ifdef IL_DEBUG_CLASS
  return i0.index() == i1.index() && i0.hash() == i1.hash();
#else
  return i0.index() == i1.index();
#endif
}

inline bool operator!=(il::spot_t i0, il::spot_t i1) {
#ifdef IL_DEBUG_CLASS
  return i0.index() != i1.index() || i0.hash() != i1.hash();
#else
  return i0.index() != i1.index();
#endif
}

}  // namespace il

#endif  // IL_SPOT_T_H
