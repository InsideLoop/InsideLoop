//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

#include <cstdio>

#include <il/SparseMatrixCSR.h>
#include <il/Timer.h>
#include <il/linear_algebra/dense/blas/dot.h>
#include <il/linear_algebra/sparse/blas/sparse_blas.h>

#include <il/linear_algebra/sparse/factorization/_test/matrix/heat.h>

int main() {
  const il::int_t nb_iteration = 10;
  const float tolerance = 1.0e-10;
  const il::int_t side = 300;

  il::SparseMatrixCSR<int, float> A = il::heat_3d<int, float>(side);
  const il::int_t n = A.size(0);
  il::Array<float> y{n, 1.0f};
  il::Array<float> x{n, 0.0f};
  il::Timer timer{};

  // This routine solves the conjugate gradient method that works for positive
  // definite symmetric matrices. The implementation comes from the wikipedia
  // page.

  timer.start();
  il::SparseMatrixBlas<int, float> A_blas{il::io, A};
  A_blas.set_nb_matrix_vector(nb_iteration);

  // r = y - A.x
  il::Array<float> r = y;
  il::blas(-1.0f, A_blas, x, 1.0f, il::io, r);
  il::Array<float> p = r;
  // delta = r.r
  float delta = il::dot(r, r);
  il::Array<float> Ap{n};

  for (il::int_t i = 0; i < nb_iteration; ++i) {
    // Ap = A.p
    il::blas(1.0f, A_blas, p, 0.0f, il::io, Ap);
    const float alpha = delta / il::dot(p, Ap);
    // x += alpha.p
    il::blas(alpha, p, 1.0f, il::io, x);
    // r -= alpha.Ap
    il::blas(-alpha, Ap, 1.0f, il::io, r);
    const float beta = il::dot(r, r);
    std::printf("Iteration: %3li,   Error: %7.3e\n", i, std::sqrt(beta));
    if (beta < tolerance * tolerance) {
      break;
    }
    // p = r + (beta / delta) * p
    il::blas(1.0f, r, beta / delta, il::io, p);
    delta = beta;
  }
  timer.stop();
  std::printf("         Time for Conjugate Gradient: %7.3f s\n", timer.time());

  return 0;
}
