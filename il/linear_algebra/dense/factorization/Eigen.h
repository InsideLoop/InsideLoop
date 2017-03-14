//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_EIGEN_H
#define IL_EIGEN_H

#include <complex>

#include <il/container/1d/Array.h>
#include <il/container/2d/Array2D.h>
#include <il/Status.h>
#include <il/linear_algebra/dense/norm.h>

#ifdef IL_MKL
#include <mkl_lapacke.h>
#elif IL_OPENBLAS
#include <OpenBLAS/lapacke.h>
#endif

namespace il {

template <typename MatrixType>
class Eigen {};

template <>
class Eigen<il::Array2D<double>> {
 private:
  il::Array<double> eigen_value_;
  il::Array<double> eigen_value_r_;
  il::Array<double> eigen_value_i_;

 public:
  // Compute eigen values of A
  Eigen(il::Array2D<double> A, il::io_t, il::Status& status);

  // Get the Eigen values
  // - The precision looks bad if the matrix can't be diagonalized in C
  il::Array<std::complex<double>> eigen_value() const;
};

Eigen<il::Array2D<double>>::Eigen(il::Array2D<double> A, il::io_t,
                                  il::Status& status)
    : eigen_value_{}, eigen_value_r_{}, eigen_value_i_{} {
  IL_EXPECT_FAST(A.size(0) > 0);
  IL_EXPECT_FAST(A.size(1) > 0);
  IL_EXPECT_FAST(A.size(0) == A.size(1));

  const int layout = LAPACK_COL_MAJOR;
  const lapack_int n = static_cast<lapack_int>(A.size(0));
  const lapack_int ilo = 1;
  const lapack_int ihi = n;
  const lapack_int lda = static_cast<lapack_int>(A.capacity(0));
  il::Array<double> tau{n > 1 ? (n - 1) : 1};
  lapack_int lapack_error =
      LAPACKE_dgehrd(layout, n, ilo, ihi, A.data(), lda, tau.data());
  IL_EXPECT_FAST(lapack_error == 0);

  const char job = 'E';
  const char compz = 'N';
  const lapack_int ldz = 1;
  il::Array<double> z{ldz * n};
  il::Array<double> w{n};
  il::Array<double> wr{n};
  il::Array<double> wi{n};
  lapack_error = LAPACKE_dhseqr(layout, job, compz, n, ilo, ihi, A.data(), lda,
                                wr.data(), wi.data(), z.data(), ldz);

  IL_EXPECT_FAST(lapack_error >= 0);
  if (lapack_error == 0) {
    status.set_ok();
    eigen_value_ = std::move(w);
    eigen_value_r_ = std::move(wr);
    eigen_value_i_ = std::move(wi);
  } else {
    status.set(il::Error::matrix_eigenvalue_no_convergence);
  }
}

il::Array<std::complex<double>> Eigen<il::Array2D<double>>::eigen_value()
    const {
  il::Array<std::complex<double>> ans{eigen_value_r_.size()};

  for (il::int_t i = 0; i < ans.size(); ++i) {
    ans[i] = std::complex<double>{eigen_value_r_[i], eigen_value_i_[i]};
  }

  return ans;
}
}

#endif  // IL_EIGEN_H
