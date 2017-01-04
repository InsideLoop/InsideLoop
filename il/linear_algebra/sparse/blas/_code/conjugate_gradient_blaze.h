//==============================================================================
//
//                                  InsideLoop
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.txt for details.
//
//==============================================================================

void code() {
  const il::int_t nb_iteration = 3000;
  const double tolerance = 0.0;

  const il::int_t side = 70;
  il::SparseMatrixCSR<int, double> A = il::heat_3d<int>(side);
  const il::int_t n = A.size(0);
  il::Array<double> y{n, 1.0};
  il::Array<double> x{n, 0.0};
  il::Timer timer{};

  const std::size_t iterations = nb_iteration;
  const double tolerance = 0.0;

  blaze::CompressedMatrix<double, blaze::rowMajor> A_cl(n, n);
  blaze::DynamicVector<double, blaze::columnVector> x_cl(n, 0.0), b_cl(n, 1.0),
      r_cl(n), p_cl(n), Ap_cl(n);
  double alpha_cl, beta_cl, delta_cl;
  for (il::int_t i = 0; i < A.size(0); ++i) {
    for (il::int_t k = A.row(i); k < A.row(i + 1); ++k) {
      A_cl(i, A.column(k)) = A.element(k);
    }
  }

  // Performing the CG algorithm

  timer.start();
  r_cl = b_cl - A_cl * x_cl;
  p_cl = r_cl;
  delta_cl = (r_cl, r_cl);

  for (std::size_t iteration = 0UL; iteration < iterations; ++iteration) {
    Ap_cl = A_cl * p_cl;
    alpha_cl = delta_cl / (p_cl, Ap_cl);
    x_cl += alpha_cl * p_cl;
    r_cl -= alpha_cl * Ap_cl;
    beta_cl = (r_cl, r_cl);
    if (std::sqrt(beta_cl) < tolerance) {
      std::printf("Iteration: %3lu,   Error: %7.3e\n", iteration,
                  std::sqrt(beta_cl));
      break;
    }
    p_cl = r_cl + (beta_cl / delta_cl) * p_cl;
    delta_cl = beta_cl;
  }
  timer.stop();
  std::printf("    Time for Conjugate Gradient Blaze: %7.3f s\n", timer.elapsed());
};
