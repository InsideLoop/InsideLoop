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

#pragma once

#ifdef IL_MKL

#include "mkl_blas.h"
#include "mkl_rci.h"
#include "mkl_service.h"
#include "mkl_spblas.h"

#include <il/Array.h>
#include <il/StaticArray.h>
#include <il/math.h>

namespace il {

template <typename M>
class MatrixFreeGmres {
 private:
  il::int_t max_nb_iteration_;
  il::int_t restart_iteration_;
  double relative_precision_;
  il::int_t nb_iteration_;
  il::StaticArray<int, 128> ipar_;
  il::StaticArray<double, 128> dpar_;

 public:
  MatrixFreeGmres();
  MatrixFreeGmres(double relative_precision, int max_nb_iteration,
                  int restart_iteration);
  il::Array<double> Solve(const M& m, const il::Array<double>& y);
  il::int_t nbIterations() const;
};

template <typename M>
MatrixFreeGmres<M>::MatrixFreeGmres() : MatrixFreeGmres{1.0e-3, 100, 20} {}

template <typename M>
MatrixFreeGmres<M>::MatrixFreeGmres(double relative_precision,
                                    int max_nb_iteration, int restart_iteration)
    : ipar_{}, dpar_{} {
  relative_precision_ = relative_precision;
  max_nb_iteration_ = max_nb_iteration;
  restart_iteration_ = restart_iteration;
}

template <typename M>
il::Array<double> il::MatrixFreeGmres<M>::Solve(const M& m,
                                                const il::Array<double>& y) {
  IL_EXPECT_FAST(m.size(0) == y.size());

  const int n = static_cast<int>(m.size(0));

  il::Array<double> yloc = y;
  il::Array<double> ycopy = y;
  il::Array<double> x{n, 0.0};

  const il::int_t tmp_size = (2 * restart_iteration_ + 1) * n +
                             restart_iteration_ * (restart_iteration_ + 9) / 2 +
                             1;
  il::Array<double> tmp{tmp_size};
  il::Array<double> residual{n};
  il::Array<double> trvec{n};

  int itercount = 0;
  int RCI_request;

  const char l_char = 'L';
  const char n_char = 'N';
  const char u_char = 'U';
  const int one_int = 1;
  const double minus_one_double = -1.0;

  dfgmres_init(&n, x.data(), yloc.data(), &RCI_request, ipar_.data(),
               dpar_.data(), tmp.data());
  IL_EXPECT_FAST(RCI_request == 0);

  // ipar_[0]: The size of the matrix
  // ipar_[1]: The default value is 6 and specifies that the errors are reported
  //           to the screen
  // ipar_[2]: Contains the current stage of the computation. The initial value
  //           is 1
  // ipar_[3]: Contains the current iteration number. The initial value is 0
  // ipar_[4]: Specifies the maximum number of iterations. The default value is
  //           min(150, n)
  ipar_[4] = static_cast<int>(max_nb_iteration_);
  // ipar_[5]: If the value is not 0, is report error messages according to
  //           ipar_[1]. The default value is 1.
  // ipar_[6]: For Warnings.
  // ipar_[7]: If the value is not equal to 0, the dfgmres performs the stopping
  //           test ipar_[3] <= ipar 4. The default value is 1.
  // ipar_[8]: If the value is not equal to 0, the dfgmres performs the residual
  //           stopping test dpar_[4] <= dpar_[3]. The default value is 1.
  // ipar_[9]: For a used-defined stopping test
  // ipar_[10]: If the value is set to 0, the routine runs the
  //            non-preconditionned version of the algorithm. The default value
  //            is 0.
  ipar_[10] = 0;
  // ipar_[13]: Contains the internal iteration counter that counts the number
  //            of iterations before the restart takes place. The initial value
  //            is 0.
  // ipar_[14]: Specifies the number of the non-restarted FGMRES iterations.
  //            To run the restarted version of the FGMRES method, assign the
  //            number of iterations to ipar_[14] before the restart.
  //            The default value is min(150, n) which means that by default,
  //            the non-restarted version of FGMRES method is used.
  ipar_[14] = static_cast<int>(restart_iteration_);
  // ipar_[30] = behaviour_zero_diagonal_;

  // dpar_[0]: Specifies the relative tolerance. The default value is 1.0e-6
  dpar_[0] = 1.0e-6;
  // dpar_[1]: Specifies the absolute tolerance. The default value is 1.0
  // dpar_[2]: Specifies the Euclidean norm of the initial residual. The initial
  //          value is 0.0
  // dpar_[3]: dpar_[0] * dpar_[2] + dpar_[1]
  //          The value for the residual under which the iteration is stopped
  //          (?)
  // dpar_[4]: Specifies the euclidean norm of the current residual.
  // dpar_[5]: Specifies the euclidean norm of the previsous residual
  // dpar_[6]: Contains the norm of the generated vector
  // dpar_[7]: Contains the tolerance for the norm of the generated vector
  // dpar_[30] = zero_diagonal_threshold_;
  // dpar_[31] = zero_diagonal_replacement_;

  dfgmres_check(&n, x.data(), yloc.data(), &RCI_request, ipar_.data(),
                dpar_.data(), tmp.data());
  IL_EXPECT_FAST(RCI_request == 0);
  bool stop_iteration = false;
  double y_norm = dnrm2(&n, yloc.data(), &one_int);
  while (!stop_iteration) {
    // The beginning of the iteration
    dfgmres(&n, x.data(), yloc.data(), &RCI_request, ipar_.data(), dpar_.data(),
            tmp.data());
    switch (RCI_request) {
      case 0:
        // In that case, the solution has been found with the right precision.
        // This occurs only if the stopping test is fully automatic.
        stop_iteration = true;
        break;
      case 1: {
        // This is a Free Matrix/Vector multiplication
        // It takes the input from tmp[ipar_[21]] and put it into tmp[ipar_[22]]
        il::ArrayView<double> my_x{tmp.data() + ipar_[21] - 1, n};
        il::ArrayEdit<double> my_y{tmp.data() + ipar_[22] - 1, n};
        for (il::int_t i = 0; i < n; ++i) {
          my_y[i] = 0.0;
        }
        m.dot(my_x, il::io, my_y);
      } break;
      case 2: {
        ipar_[12] = 1;
        // Retrieve iteration number AND update sol
        dfgmres_get(&n, x.data(), ycopy.data(), &RCI_request, ipar_.data(),
                    dpar_.data(), tmp.data(), &itercount);
        // Compute the current true residual via MKL (Sparse) BLAS
        // routines. It multiplies the matrix A with yCopy and
        // store the result in residual.
        //--------------- To Change
        il::ArrayView<double> my_x = ycopy.view();
        il::ArrayEdit<double> my_y = residual.Edit();
        for (il::int_t i = 0; i < n; ++i) {
          my_y[i] = 0.0;
        }
        m.dot(my_x, il::io, my_y);
        // Compute: residual = A.(current x) - y
        // Note that A.(current x) is stored in residual before this operation
        daxpy(&n, &minus_one_double, yloc.data(), &one_int, residual.data(),
              &one_int);
        // This number plays a critical role in the precision of the method
        double norm_residual = dnrm2(&n, residual.data(), &one_int);
        double error_ratio = norm_residual / y_norm;
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
        il::ArrayView<double> my_x{tmp.data() + ipar_[21] - 1, n};
        il::ArrayEdit<double> my_y{tmp.data() + ipar_[22] - 1, n};
        for (il::int_t i = 0; i < n; ++i) {
          my_y[i] = my_x[i];
        }
      } break;
      case 4:
        // If RCI_REQUEST=4, then check if the norm of the next
        // generated vector is not zero up to rounding and
        // computational errors. The norm is contained in DPAR(7)
        // parameter
        if (dpar_[6] == 0.0) {
          stop_iteration = true;
        }
        break;
      default:
        IL_EXPECT_FAST(false);
        break;
    }
  }
  ipar_[12] = 0;
  dfgmres_get(&n, x.data(), yloc.data(), &RCI_request, ipar_.data(),
              dpar_.data(), tmp.data(), &itercount);

  nb_iteration_ = itercount;

  return x;
}

template <typename M>
il::int_t il::MatrixFreeGmres<M>::nbIterations() const {
  return nb_iteration_;
}

}  // namespace il

#endif  // IL_MKL
