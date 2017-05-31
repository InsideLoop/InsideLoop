//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_SOBOL_H
#define IL_SOBOL_H

#include <il/Array2C.h>

#ifdef IL_BLAS
#include <mkl.h>
#endif

namespace il {

inline il::Array2C<double> sobol(il::int_t nb_point, il::int_t dim, double a,
                                 double b) {
  il::Array2C<double> A{nb_point, dim};

  VSLStreamStatePtr stream;
  const MKL_INT brng = VSL_BRNG_SOBOL;
  int error_code;
  error_code = vslNewStream(&stream, brng, static_cast<MKL_INT>(dim));
  error_code = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, dim * nb_point,
                            A.data(), a, b);
  error_code = vslDeleteStream(&stream);

  return A;
}

}  // namespace il

#endif  // IL_SOBOL_H
