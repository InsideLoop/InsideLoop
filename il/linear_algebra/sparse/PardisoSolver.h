//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#ifndef IL_PARDISOSOLVER_H
#define IL_PARDISOSOLVER_H

#include <chrono>
#include <iostream>

#include <il/container/1d/Array.h>
#include <il/linear_algebra/sparse/container/SparseArray2C.h>

#include <mkl.h>

namespace il {

class PardisoSolver {
  static_assert(sizeof(il::int_t) == sizeof(int),
                "il::PardisoSolver: works only with 32 bit integers");

 private:
  int n_;
  int pardiso_nrhs_;
  int pardiso_max_fact_;
  int pardiso_mnum_;
  int pardiso_mtype_;
  int pardiso_msglvl_;
  int pardiso_iparm_[64];
  void *pardiso_pt_[64];
  bool is_symbolic_factorization_;
  bool is_numerical_factorization_;
  const double *matrix_element_;

 public:
  PardisoSolver();
  ~PardisoSolver();
  void symbolic_factorization(const il::SparseArray2C<double> &A);
  void numerical_factorization(const il::SparseArray2C<double> &A);
  il::Array<double> solve(const il::SparseArray2C<double> &A,
                          const il::Array<double> &y);
  il::Array<double> solve_iterative(const il::SparseArray2C<double> &A,
                                    const il::Array<double> &y);

 private:
  void release();
};

inline PardisoSolver::PardisoSolver() {
  n_ = 0;
  pardiso_nrhs_ = 1;

  // This is used to store multiple LU factorization using the same sparsity
  // pattern
  pardiso_max_fact_ = 1;
  pardiso_mnum_ = 1;

  // The following matrices are accepted
  // - 1: real and structurally symmetric
  // - 2: real and symmetric positive definite
  // - -2: real and symmetric indefinite
  // - 3: complex and structurally symmetric
  // - 4: complex and hermitian positive definite
  // - -4: complex and hermitian indefinite
  // - 6: complex and symmetric
  // - 11: real and nonsymmetric
  // - 13: complex and nonsymmetric
  pardiso_mtype_ = 11;

  pardiso_msglvl_ = 0;

  for (int i = 0; i < 64; ++i) {
    pardiso_iparm_[i] = 0;
  }

  // Default values
  // - 0: use default values
  // - 1: use values given in iparm
  pardiso_iparm_[0] = 1;

  // Fill-in reducing algorithm for the input matrix
  // - 0: use the minimum degree algorithm
  // - 2: use the METIS reordering scheme
  // - 3: use the parallel METIS reordering scheme. It can decrease the chrono
  // of
  //      computation on multicore computers, especially when the phase 1 takes
  //      significant chrono.
  pardiso_iparm_[1] = 2;

  // Although Pardiso is a direct solver, it allows you to be used as an
  // iterative solver using a previously computed LU factorization
  // - 0: use it as a direct solver
  // - 10 * L + K: Use an itera1tive solver with
  //   - K = 0: The direct solver is used
  //   - K = 1: CGS replaces the direct solver preconditionned by a previous LU
  //   - K = 2: CGS replaces the direct solver preconditionned by a previous
  //            Cholesky factorization
  //   - L: The Krylov subspace iteration is stopped when |dxi|/|dx0| <= 10^(-L)
  pardiso_iparm_[3] = 0;

  // Permutation
  // - 0: don't use a user permutation
  pardiso_iparm_[4] = 0;

  // Where to write the solution of A.x = y
  // - 0: write the array on x
  // - 1: overwrite the solution on y
  pardiso_iparm_[5] = 0;

  // Iterative refinement step
  // - 0: The solver uses two steps of iterative refinement when a perturbated
  //      pivot is used during the factorization
  // - n > 0: Performs at most n iterative refinements
  // - -n < 0: Performs at most n iterative refinements but the accumulation
  //   of the residual is computed using extended precision
  //
  // The number of iterations used is stored in iparm[6]
  pardiso_iparm_[7] = 0;

  // This parameter instructs pardiso_64 how to handle small pivots or zero
  // pivots
  // for unsymmetric matrices and symmetric matrices. Here, we use pivoting
  // perturbation of 1.0e-13.
  pardiso_iparm_[9] = 13;

  // Scaling
  // 0 : No Scaling vector
  // 1 : Scale the matrix so that the diagonal elements are equal to 1 and the
  //     absolute values of the off-diagonal entries are less or equal to 1.
  //     Note that in the analysis phase, you need to supply the numerical
  //     values of A in case of scaling.
  pardiso_iparm_[10] = 1;

  // Specify which system to solve
  // - 0: solve the problem A.x = b
  // - 1: solve the conjugate transpose problem AH.x = b based upon the
  //      factorization of A
  // - 2: solve the transpose problem AT.x = b based upon the factorization of A
  pardiso_iparm_[11] = 0;

  // improved accuracy using (non-)symmetric weighted  matchings
  // - 0: disable matching
  // - 1: enable matching, which is the default for nonsymmetric matrices
  //      In this case, you need to provide the values of A during the symbolic
  //      factorization
  pardiso_iparm_[12] = 1;

  // Report the number of nonzeros in the factors
  // - n < 0: enable report, -1 being the default values.
  // - n >= 0: disable the report
  pardiso_iparm_[17] = -1;

  // Report the number of MFLOPS necessary to factor the matrix A (it increases
  // the computation chrono)
  // - n < 0: enable report, -1 being the default values.
  // - n >= 0: disable the report
  pardiso_iparm_[18] = 0;

  // pivoting for symmetric indefinite matrices (useless for unsymmetric
  // matrices)
  pardiso_iparm_[20] = 0;

  // Parallel factorization control (new version)
  // - 0: uses the classic algorithm for factorization
  // - 1: uses a two-level factorization algorithm which generaly improves
  //      scalability in case of parallel factorization with many threads
  pardiso_iparm_[23] = 1;

  // parallel forward/backward solve control
  // - 0: uses a parallel algorithm for the forward/backward substitution
  // - 1: uses a serial algorithm for the forward/backward substitution
  pardiso_iparm_[24] = 0;

  // Matrix checker
  // - 0: does not check the sparse matrix representation
  // - 1: checks the sparse matrix representation. It checks if column are in
  //      increasing order in each row
  pardiso_iparm_[26] = 0;

  // Input/Output and working precision
  // - 0: double precision
  // - 1: single precision
  pardiso_iparm_[27] = 0;

  // Sparsity of the second member
  // - 0: don't expect a sparse second member
  // - Check the MKL documentation for other values
  pardiso_iparm_[30] = 0;

  // Type of indexing for the columns and the rows
  // - 0: Use Fortran indexing (starting at 1)
  // - 1: Use C indexing (starting at 0)
  pardiso_iparm_[34] = 1;

  // In-core / Out-of-core pardiso_64
  // - 0: Use In-core pardiso_64
  // - 1: Use Out-of-core pardiso_64. Use only if you need to solver very large
  //      problems that do not fit in memory. It uses the hard-drive to store
  //      the elements.
  pardiso_iparm_[59] = 0;

  for (int i = 0; i < 64; ++i) {
    pardiso_pt_[i] = nullptr;
  }

  is_symbolic_factorization_ = false;
  is_numerical_factorization_ = false;
  matrix_element_ = nullptr;
}

inline PardisoSolver::~PardisoSolver() { release(); }

void PardisoSolver::symbolic_factorization(const il::SparseArray2C<double> &A) {
  IL_ASSERT(A.size(0) == A.size(1));
  n_ = A.size(0);

  const int phase = 11;
  int error = 0;
  int i_dummy;

  release();
  pardiso(pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_,
          &phase, &n_, A.element_data(), A.row_data(), A.column_data(),
          &i_dummy, &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_, nullptr,
          nullptr, &error);
  IL_ASSERT(error == 0);

  is_symbolic_factorization_ = true;
  matrix_element_ = A.element_data();
}

void PardisoSolver::numerical_factorization(
    const il::SparseArray2C<double> &A) {
  IL_ASSERT(matrix_element_ = A.element_data());
  IL_ASSERT(is_symbolic_factorization_);
  IL_ASSERT(A.size(0) == n_);
  IL_ASSERT(A.size(1) == n_);

  const int phase = 22;
  int error = 0;
  int i_dummy;

  pardiso(pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_,
          &phase, &n_, A.element_data(), A.row_data(), A.column_data(),
          &i_dummy, &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_, nullptr,
          nullptr, &error);
  IL_ASSERT(error == 0);

  is_numerical_factorization_ = true;
}

inline il::Array<double> PardisoSolver::solve(
    const il::SparseArray2C<double> &A, const il::Array<double> &y) {
  IL_ASSERT(matrix_element_ = A.element_data());
  IL_ASSERT(is_numerical_factorization_);
  IL_ASSERT(A.size(0) == n_);
  IL_ASSERT(A.size(1) == n_);
  IL_ASSERT(y.size() == n_);
  il::Array<double> x{n_};

  const int phase = 33;
  int error = 0;
  int i_dummy;
  pardiso(pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_,
          &phase, &n_, A.element_data(), A.row_data(), A.column_data(),
          &i_dummy, &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_,
          const_cast<double *>(y.data()), x.data(), &error);

  IL_ASSERT(error == 0);

  return x;
}

inline il::Array<double> PardisoSolver::solve_iterative(
    const il::SparseArray2C<double> &A, const il::Array<double> &y) {
  IL_ASSERT(matrix_element_ = A.element_data());
  IL_ASSERT(is_numerical_factorization_);
  IL_ASSERT(A.size(0) == n_);
  IL_ASSERT(A.size(1) == n_);
  IL_ASSERT(y.size() == n_);
  il::Array<double> x{n_};

  const int old_solver = pardiso_iparm_[3];
  // 6 digits of accuracy using LU decomposition
  pardiso_iparm_[3] = 21;

  const int phase = 33;
  int error = 0;
  int i_dummy;
  pardiso(pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_,
          &phase, &n_, A.element_data(), A.row_data(), A.column_data(),
          &i_dummy, &pardiso_nrhs_, pardiso_iparm_, &pardiso_msglvl_,
          const_cast<double *>(y.data()), x.data(), &error);
  IL_ASSERT(error == 0);

  pardiso_iparm_[3] = old_solver;

  return x;
}

inline void PardisoSolver::release() {
  const int phase = -1;
  int error = 0;
  int i_dummy;
  if (is_symbolic_factorization_) {
    PARDISO(pardiso_pt_, &pardiso_max_fact_, &pardiso_mnum_, &pardiso_mtype_,
            &phase, &n_, nullptr, nullptr, nullptr, &i_dummy, &pardiso_nrhs_,
            pardiso_iparm_, &pardiso_msglvl_, nullptr, nullptr, &error);
    IL_ASSERT(error == 0);

    is_symbolic_factorization_ = false;
    is_numerical_factorization_ = false;
  }
}
}

#endif  // IL_PARDISOSOLVER_H
