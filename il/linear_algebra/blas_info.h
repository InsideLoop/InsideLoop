//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_BLAS_INFO_H
#define IL_BLAS_INFO_H

namespace il {

enum class Blas {
  regular,
  transpose,
  conjugate_transpose,
  symmetric_upper,
  symmetric_lower,
  hermitian_upper,
  hermitian_lower
};

}

#endif  // IL_BLAS_INFO_H
