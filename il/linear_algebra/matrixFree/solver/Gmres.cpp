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

#include "Gmres.h"

#include <il/Array.h>

#include "mkl_blas.h"
#include "mkl_rci.h"

namespace il {

Gmres::Gmres(double relative_precision, il::int_t max_nb_iterations,
             il::int_t restart_iteration) {
  IL_EXPECT_MEDIUM(relative_precision >= 0.0);
  IL_EXPECT_MEDIUM(max_nb_iterations >= 0);
  IL_EXPECT_MEDIUM(restart_iteration >= 0);

  relative_precision_ = relative_precision;
  max_nb_iterations_ = max_nb_iterations;
  restart_iteration_ = restart_iteration;
  nb_iterations_ = -1;
}

void Gmres::solve(const ArrayFunctor<double>& a, const ArrayFunctor<double>& b,
                  il::ArrayView<double> y, bool use_preconditionner,
                  bool use_x_as_initial_value, il::io_t,
                  il::ArrayEdit<double> x) {
  IL_EXPECT_FAST(a.sizeInput() == x.size());
  IL_EXPECT_FAST(a.sizeOutput() == y.size());
  IL_EXPECT_FAST(a.sizeInput() == b.sizeOutput());
  IL_EXPECT_FAST(a.sizeOutput() == b.sizeInput());
  IL_EXPECT_FAST(a.sizeInput() == a.sizeOutput());

  const MKL_INT n = static_cast<MKL_INT>(a.sizeInput());

  il::Array<double> y_local{y};
  il::Array<double> y_copy = y_local;

  // Setup the initial solution to x if it should be used as an initial value
  // or to 0.0 in case it should not
  il::Array<double> x_local{n};
  if (use_x_as_initial_value) {
    for (il::int_t i = 0; i < n; ++i) {
      x_local[i] = x[i];
    }
  } else {
    for (il::int_t i = 0; i < n; ++i) {
      x_local[i] = 0.0;
    }
  }

  const il::int_t tmp_size = (2 * restart_iteration_ + 1) * n +
                             restart_iteration_ * (restart_iteration_ + 9) / 2 +
                             1;
  il::Array<double> tmp{tmp_size};
  il::Array<double> residual{n};
  il::Array<double> trvec{n};

  int itercount = 0;
  MKL_INT RCI_request;
  const char l_char = 'L';
  const char n_char = 'N';
  const char u_char = 'U';
  const int one_int = 1;
  const double minus_one_double = -1.0;

  il::StaticArray<MKL_INT, 128> ipar;
  il::StaticArray<double, 128> dpar;

  dfgmres_init(&n, x_local.data(), y_local.data(), &RCI_request, ipar.data(),
               dpar.data(), tmp.data());
  IL_EXPECT_FAST(RCI_request == 0);

  // ipar[0]: The size of the matrix
  // ipar[1]: The default value is 6 and specifies that the errors are reported
  //           to the screen
  // ipar[2]: Contains the current stage of the computation. The initial value
  //           is 1
  // ipar[3]: Contains the current iteration number. The initial value is 0
  // ipar[4]: Specifies the maximum number of iterations. The default value is
  //           min(150, n)
  ipar[4] = static_cast<MKL_INT>(max_nb_iterations_);
  // ipar[5]: If the value is not 0, is report error messages according to
  //           ipar[1]. The default value is 1.
  // ipar[6]: For Warnings.
  // ipar[7]: If the value is not equal to 0, the dfgmres performs the stopping
  //           test ipar[3] <= ipar 4. The default value is 1.
  // ipar[8]: If the value is not equal to 0, the dfgmres performs the residual
  //           stopping test dpar[4] <= dpar[3]. The default value is 1.
  // ipar[9]: For a used-defined stopping test
  // ipar[10]: If the value is set to 0, the routine runs the
  //            non-preconditionned version of the algorithm. The default value
  //            is 0.
  ipar[10] = use_preconditionner ? 1 : 0;
  // ipar[13]: Contains the internal iteration counter that counts the number
  //            of iterations before the restart takes place. The initial value
  //            is 0.
  // ipar[14]: Specifies the number of the non-restarted FGMRES iterations.
  //            To run the restarted version of the FGMRES method, assign the
  //            number of iterations to ipar[14] before the restart.
  //            The default value is min(150, n) which means that by default,
  //            the non-restarted version of FGMRES method is used.
  ipar[14] = static_cast<MKL_INT>(restart_iteration_);
  // ipar[30] = behaviour_zero_diagonal_;

  // dpar[0]: Specifies the relative tolerance. The default value is 1.0e-6
  dpar[0] = relative_precision_;
  // dpar[1]: Specifies the absolute tolerance. The default value is 1.0
  // dpar[2]: Specifies the Euclidean norm of the initial residual. The initial
  //          value is 0.0
  // dpar[3]: dpar[0] * dpar[2] + dpar[1]
  //          The value for the residual under which the iteration is stopped
  //          (?)
  // dpar[4]: Specifies the euclidean norm of the current residual.
  // dpar[5]: Specifies the euclidean norm of the previsous residual
  // dpar[6]: Contains the norm of the generated vector
  // dpar[7]: Contains the tolerance for the norm of the generated vector
  // dpar[30] = zero_diagonal_threshold_;
  // dpar[31] = zero_diagonal_replacement_;

  dfgmres_check(&n, x_local.data(), y_local.data(), &RCI_request, ipar.data(),
                dpar.data(), tmp.data());
  IL_EXPECT_FAST(RCI_request == 0);

  bool stop_iteration = false;
  double y_norm = dnrm2(&n, y_local.data(), &one_int);
  while (!stop_iteration) {
    // The beginning of the iteration
    dfgmres(&n, x_local.data(), y_local.data(), &RCI_request, ipar.data(),
            dpar.data(), tmp.data());
    switch (RCI_request) {
      case 0:
        // In that case, the solution has been found with the right precision.
        // This occurs only if the stopping test is fully automatic.
        stop_iteration = true;
        break;
      case 1: {
        // This is a Free Matrix/Vector multiplication
        // It takes the input from tmp[ipar[21]] and put it into tmp[ipar[22]]
        il::ArrayView<double> x_view{tmp.data() + ipar[21] - 1, n};
        il::ArrayEdit<double> y_edit{tmp.data() + ipar[22] - 1, n};
        a(x_view, il::io, y_edit);
      } break;
      case 2: {
        ipar[12] = 1;
        // Retrieve iteration number AND update sol
        dfgmres_get(&n, x_local.data(), y_copy.data(), &RCI_request,
                    ipar.data(), dpar.data(), tmp.data(), &itercount);
        // Compute the current true residual via MKL (Sparse) BLAS
        // routines. It multiplies the matrix A with yCopy and
        // store the result in residual.
        //--------------- To Change
        il::ArrayView<double> y_copy_view = y_copy.view();
        il::ArrayEdit<double> residual_edit = residual.edit();
        a(y_copy_view, il::io, residual_edit);
        // Compute: residual = A.(current x_local) - y
        // Note that A.(current x_local) is stored in residual before this
        // operation
        daxpy(&n, &minus_one_double, y_local.data(), &one_int, residual.data(),
              &one_int);
        // This number plays a critical role in the precision of the method
        double norm_residual = dnrm2(&n, residual.data(), &one_int);
        //        double error_ratio = norm_residual / y_norm;
        if (norm_residual <= relative_precision_ * y_norm) {
          stop_iteration = true;
        }
      } break;
      case 3: {
        // If RCI_REQUEST=3, then apply the preconditioner on the
        // vector TMP(IPAR(22)) and put the result in vector
        // TMP(IPAR(23)). Here is the recommended usage of the
        // result produced by ILUT routine via standard MKL Sparse
        // Blas solver rout'ine mkl_dcsrtrsv
        il::ArrayView<double> x_view{tmp.data() + ipar[21] - 1, n};
        il::ArrayEdit<double> y_edit{tmp.data() + ipar[22] - 1, n};
        b(x_view, il::io, y_edit);
      } break;
      case 4:
        // If RCI_REQUEST=4, then check if the norm of the next
        // generated vector is not zero up to rounding and
        // computational errors. The norm is contained in DPAR(7)
        // parameter
        if (dpar[6] == 0.0) {
          stop_iteration = true;
        }
        break;
      default:
        IL_EXPECT_FAST(false);
        break;
    }
  }
  ipar[12] = 0;
  dfgmres_get(&n, x_local.data(), y_local.data(), &RCI_request, ipar.data(),
              dpar.data(), tmp.data(), &itercount);

  nb_iterations_ = itercount;

  for (il::int_t i = 0; i < n; ++i) {
    x[i] = x_local[i];
  }
}

il::int_t Gmres::nbIterations() const {
  IL_EXPECT_MEDIUM(nb_iterations_ >= 0);

  return nb_iterations_;
}

}  // namespace il
