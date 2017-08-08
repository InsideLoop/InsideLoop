//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_MEMORY_H
#define IL_MEMORY_H

#include <il/Array2D.h>

namespace il {

void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

void clobber() { asm volatile("" : : : "memory"); }

// template <typename T>
// void commit_memory(il::io_t, il::Array2D<T>& A) {
//   il::int_t stride{static_cast<il::int_t>(il::page / sizeof(T))};
//   T* const data{A.data()};
//   for (il::int_t k = 0; k < A.capacity(0) * A.capacity(1); k += stride) {
//     data[k] = T{};
//   }
// }

// template <typename T>
// void warm_cache(il::io_t, il::Array2D<T>& A) {
//   T* const data{A.data()};
//   for (il::int_t k = 0; k < A.capacity(0) * A.capacity(1); ++k) {
//     data[k] = T{};
//   }
// }
}  // namespace il

#endif  // IL_MEMORY_H
