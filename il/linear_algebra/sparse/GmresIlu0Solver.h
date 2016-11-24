// The MIT License (MIT)
//
// Copyright (c) <2015> <Francois Fayard, fayard@insideloop.io>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <iostream>
#include <chrono>

#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_rci.h"
#include "mkl_service.h"

#include <il/math>
#include <il/container/1d/StaticArray.h>
#include <il/container/2d/SparseArray2D.h>

namespace il {

class GmresIlu0Solver {
 private:
  MKL_INT n_;
  il::int_t max_nb_iterations_;
  il::Array<double> bilu0_;
  bool preconditionner_computed_;

 public:
  GmresIlu0Solver(const il::SparseMatrix<double, MKL_INT>& sparse_matrix)
     ;
  il::Array<double> solve(il::SparseMatrix<double, MKL_INT>& sparse_matrix,
                           il::Array<double>& y);
};

GmresIlu0Solver::GmresIlu0Solver(const il::SparseMatrix<double, MKL_INT>& sparse_matrix)
    : n_{sparse_matrix.size(0)},
                  max_nb_iterations_{100},
                  bilu0_{sparse_matrix.nb_nonzeros()},
                  preconditionner_computed_{false} {
  IL_ASSERT(sparse_matrix.size(0) == sparse_matrix.size(1));
}

il::Array<double> il::GmresIlu0Solver::solve(
    il::SparseMatrix<double, MKL_INT>& sparse_matrix,
    il::Array<double>& y) {
  IL_ASSERT(y.size() == n_);

  auto start_time = std::chrono::high_resolution_clock::now();
  auto x = il::Array<double>{n_, 0.0};

  auto yloc = il::Array<double>{y};
  // I am not sure that ycopy is needed for the algorithm.
  // Can;t we just use y?
  auto ycopy = il::Array<double>{y};

  auto ipar = il::StaticArray<MKL_INT, 128>{0};
  ipar[14] = il::min(max_nb_iterations_, n_);  // Maximum number of iterations
  auto dpar = il::StaticArray<double, 128>{0.0};

  auto tmp = il::Array<double>{(2 * ipar[14] + 1) * n_ +
                                ipar[14] * (ipar[14] + 9) / 2 + 1};
  auto residual = il::Array<double>{n_};
  auto trvec = il::Array<double>{n_};

  //  std::cout << "Running" << std::endl;

  auto ierr = MKL_INT{0};
  auto itercount = MKL_INT{0};
  MKL_INT RCI_request;
  auto l_char = char{'L'};
  auto n_char = char{'N'};
  auto u_char = char{'U'};
  auto one_int = MKL_INT{1};
  auto minus_one_double = double{-1.0};

  dfgmres_init(&n_, x.data(), yloc.data(), &RCI_request, ipar.data(),
               dpar.data(), tmp.data());
  IL_ASSERT(RCI_request == 0);
  if (!preconditionner_computed_) {
    /*--------------------------------------------------------------------------
    // Calculate ILU0 preconditioner.
    //                      !ATTENTION!
    // DCSRILU0 routine uses some IPAR, DPAR set by DFGMRES_INIT routine.
    // Important for DCSRILU0 default entries set by DFGMRES_INIT are
    // ipar[1] = 6 - output of error messages to the screen,
    // ipar[5] = 1 - allow output of errors,
    // ipar[30]= 0 - abort DCSRILU0 calculations if routine meets zero diagonal
    //               element.
    //
    //
    // If ILU0 is going to be used out of MKL FGMRES context, than the values
    // of ipar[1], ipar[5], ipar[30], dpar[30], and dpar[31] should be user
    // provided before the DCSRILU0 routine call.
    //
    // In this example, specific for DCSRILU0 entries are set in turn:
    // ipar[30]= 1 - change small diagonal value to that given by dpar[31],
    // dpar[30]= 1.E-20 instead of the default value set by DFGMRES_INIT.
    //                  It is a small value to compare a diagonal entry with it.
    // dpar[31]= 1.E-16 instead of the default value set by DFGMRES_INIT.
    //                  It is the target value of the diagonal value if it is
    //                  small as compared to dpar[30] and the routine should
    //                  change it rather than abort DCSRILU0 calculations.
    //------------------------------------------------------------------------*/
    ipar[30] = 1;
    dpar[30] = 1.0e-20;
    dpar[31] = 1.0e-16;
    dcsrilu0(&n_, sparse_matrix.element(), sparse_matrix.row(),
             sparse_matrix.column(), bilu0_.data(), ipar.data(), dpar.data(),
             &ierr);
    IL_ASSERT(ierr == 0);
    preconditionner_computed_ = true;
  }
  //  std::cout << "Preconditionner computed" << std::endl;
  // Specifies the number of the non-restarted FGMRES iterations.
  // ipar[14] = 20;
  ipar[7] = 0;
  // Use the GMRES method with the preconditionner
  ipar[10] = 1;
  // Specifies the relative tolerance. The default value is 1.0e-6
  // dpar[0] = 1.0e-6;
  dfgmres_check(&n_, x.data(), yloc.data(), &RCI_request, ipar.data(),
                dpar.data(), tmp.data());
  IL_ASSERT(RCI_request == 0);
  auto stop_iteration = bool{false};
  auto y_norm = double{dnrm2(&n_, yloc.data(), &one_int)};
  while (!stop_iteration) {
    //    std::cout << "New iteration" << std::endl;
    // The beginning of the iteration
    dfgmres(&n_, x.data(), yloc.data(), &RCI_request, ipar.data(), dpar.data(),
            tmp.data());
    switch (RCI_request) {
      case 0:
        // In that case, the solution has been found with the right precision.
        // This occurs only if the stopping test is fully automatic.
        stop_iteration = true;
        break;
      case 1:
        // This is a Sparse matrix/Vector multiplication
        mkl_dcsrgemv(&n_char, &n_, sparse_matrix.element(), sparse_matrix.row(),
                     sparse_matrix.column(), &tmp[ipar[21] - 1],
                     &tmp[ipar[22] - 1]);
        break;
      case 2:
        ipar[12] = 1;
        // Retrieve iteration number AND update sol
        dfgmres_get(&n_, x.data(), ycopy.data(), &RCI_request, ipar.data(),
                    dpar.data(), tmp.data(), &itercount);
        // Compute the current true residual via MKL (Sparse) BLAS
        // routines. It multiplies the matrix A with yCopy and
        // store the result in residual.
        mkl_dcsrgemv(&n_char, &n_, sparse_matrix.element(), sparse_matrix.row(),
                     sparse_matrix.column(), ycopy.data(), residual.data());
        // Compute: residual = A.(current x) - y
        // Note that A.(current x) is stored in residual before this operation
        daxpy(&n_, &minus_one_double, yloc.data(), &one_int, residual.data(),
              &one_int);
        // This number plays a critical role in the precision of the method
        //      std::cout << "Residual: " << dnrm2(&n_, residual.data(),
        //      &one_int) << ",  y: " << y_norm << std::endl;
        //      std::cout << "Relative norm of residual: " << dnrm2(&n_,
        //      residual.data(), &one_int) / y_norm << std::endl;
        if (dnrm2(&n_, residual.data(), &one_int) <= 1.0e-7 * y_norm) {
          stop_iteration = true;
        }
        break;
      case 3:
        // If RCI_REQUEST=3, then apply the preconditioner on the
        // vector TMP(IPAR(22)) and put the result in vector
        // TMP(IPAR(23)). Here is the recommended usage of the
        // result produced by ILUT routine via standard MKL Sparse
        // Blas solver rout'ine mkl_dcsrtrsv
        mkl_dcsrtrsv(&l_char, &n_char, &u_char, &n_, bilu0_.data(),
                     sparse_matrix.row(), sparse_matrix.column(),
                     &tmp[ipar[21] - 1], trvec.data());
        mkl_dcsrtrsv(&u_char, &n_char, &n_char, &n_, bilu0_.data(),
                     sparse_matrix.row(), sparse_matrix.column(), trvec.data(),
                     &tmp[ipar[22] - 1]);
        break;
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
        IL_ASSERT(false);
        break;
    }
  }
  ipar[12] = 0;
  dfgmres_get(&n_, x.data(), yloc.data(), &RCI_request, ipar.data(),
              dpar.data(), tmp.data(), &itercount);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  end_time - start_time).count();
  std::cout << "Number of iterations: " << itercount << std::endl;
  std::cout << "Time: " << time / 1.0e9 << " seconds" << std::endl;

  return x;
};
}
