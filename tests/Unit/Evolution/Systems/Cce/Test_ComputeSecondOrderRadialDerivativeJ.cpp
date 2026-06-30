// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/Initialize/ComputeSecondOrderRadialDerivativeJ.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshFiltering.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace Cce::InitializeJ::CauchySecondOrder_detail {
namespace {

// A band-limited random spin-weighted boundary scalar, built the same way the
// other CCE worldtube tests build random angular data.
template <int Spin, typename Generator, typename Distribution>
Scalar<SpinWeighted<ComplexDataVector, Spin>> random_boundary_scalar(
    const gsl::not_null<Generator*> generator,
    const gsl::not_null<Distribution*> distribution, const size_t l_max) {
  SpinWeighted<ComplexModalVector, Spin> modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
  Spectral::Swsh::TestHelpers::generate_swsh_modes<Spin>(
      make_not_null(&modes.data()), generator, distribution, 1, l_max);
  Scalar<SpinWeighted<ComplexDataVector, Spin>> result{};
  get(result) = Spectral::Swsh::inverse_swsh_transform(l_max, 1, modes);
  Spectral::Swsh::filter_swsh_boundary_quantity(make_not_null(&get(result)),
                                                l_max, l_max / 2);
  return result;
}

// `compute_dy_dy_j` returns the value of dy^2 J for which the worldtube
// H-hypersurface residual vanishes. The defining property is therefore that
// feeding that value back into `evaluate_worldtube_h_residual` reproduces the
// H equation to machine precision. We also check that the residual really is
// affine in (dy^2 J, dy^2 Jbar), which is what makes the three-point probing
// inside `compute_dy_dy_j` exact.
void test_h_equation_is_satisfied(const size_t l_max) {
  CAPTURE(l_max);
  MAKE_GENERATOR(generator);
  // Moderate amplitudes keep k = sqrt(1 + |J|^2) and exp(2 beta) of order one
  // and the 2x2 conjugate system well conditioned.
  UniformCustomDistribution<double> coefficient_distribution{0.1, 0.5};
  const size_t number_of_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const ComplexDataVector zero_dy_dy_j{number_of_points, 0.0};

  const auto j = random_boundary_scalar<2>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto u = random_boundary_scalar<1>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto q = random_boundary_scalar<1>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto du_j = random_boundary_scalar<2>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto dr_j = random_boundary_scalar<2>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto du_dr_j = random_boundary_scalar<2>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  // beta, w, du_r and R are real spin-0 worldtube quantities; R is a positive
  // areal radius.
  auto beta = random_boundary_scalar<0>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  get(beta).data() = real(get(beta).data());
  auto w = random_boundary_scalar<0>(make_not_null(&generator),
                                     make_not_null(&coefficient_distribution),
                                     l_max);
  get(w).data() = real(get(w).data());
  auto du_r = random_boundary_scalar<0>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  get(du_r).data() = real(get(du_r).data());
  auto r = random_boundary_scalar<0>(make_not_null(&generator),
                                     make_not_null(&coefficient_distribution),
                                     l_max);
  get(r).data() = 5.0 + real(get(r).data());

  // `compute_dy_dy_j` takes the physical worldtube data and converts it to the
  // numerical (constant y) coordinate internally. We reproduce that conversion
  // here to drive `evaluate_worldtube_h_residual`, which works purely in the
  // numerical coordinate:
  //   dy_j           = (R / 2) Dr<J>,
  //   h    = Du<J> + Du<R> Dr<J>,
  //   dy_h = (1 / 2) (Du<R> Dr<J> + R Du<Dr<J>>).
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_j{number_of_points};
  get(dy_j).data() = 0.5 * get(r).data() * get(dr_j).data();
  Scalar<SpinWeighted<ComplexDataVector, 2>> h{number_of_points};
  get(h).data() = get(du_j).data() + get(du_r).data() * get(dr_j).data();
  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_h{number_of_points};
  get(dy_h).data() = 0.5 * (get(du_r).data() * get(dr_j).data() +
                            get(r).data() * get(du_dr_j).data());

  Scalar<SpinWeighted<ComplexDataVector, 2>> dy_dy_j{number_of_points};
  compute_dy_dy_j(make_not_null(&dy_dy_j), j, u, w, beta, q, du_j, dr_j,
                  du_dr_j, du_r, r, l_max);

  // The natural scale of the H equation is the size of its source term, the
  // residual evaluated with dy^2 J set to zero.
  const auto residual_without_dy_dy_j = evaluate_worldtube_h_residual(
      zero_dy_dy_j, j, u, w, beta, q, dy_j, h, dy_h, du_r, r, l_max);
  double scale = 0.0;
  for (const auto entry : get(residual_without_dy_dy_j).data()) {
    scale = std::max(scale, std::abs(entry));
  }

  const auto residual_at_solution = evaluate_worldtube_h_residual(
      get(dy_dy_j).data(), j, u, w, beta, q, dy_j, h, dy_h, du_r, r, l_max);
  const Approx h_equation_approx =
      Approx::custom().epsilon(1.0e-10).scale(scale);
  CHECK_ITERABLE_CUSTOM_APPROX(get(residual_at_solution).data(), zero_dy_dy_j,
                               h_equation_approx);

  // The probing in `compute_dy_dy_j` is only exact if the residual is affine in
  // (dy^2 J, dy^2 Jbar): F(a + b) = F(a) + F(b) - F(0). Verify that against two
  // independent random trial values.
  const auto trial_a = random_boundary_scalar<2>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto trial_b = random_boundary_scalar<2>(
      make_not_null(&generator), make_not_null(&coefficient_distribution),
      l_max);
  const auto evaluate = [&](const ComplexDataVector& trial) {
    return get(evaluate_worldtube_h_residual(trial, j, u, w, beta, q, dy_j, h,
                                             dy_h, du_r, r, l_max))
        .data();
  };
  const ComplexDataVector residual_of_sum =
      evaluate(get(trial_a).data() + get(trial_b).data());
  const ComplexDataVector affine_prediction = evaluate(get(trial_a).data()) +
                                              evaluate(get(trial_b).data()) -
                                              evaluate(zero_dy_dy_j);
  const Approx affine_approx = Approx::custom().epsilon(1.0e-10).scale(scale);
  CHECK_ITERABLE_CUSTOM_APPROX(residual_of_sum, affine_prediction,
                               affine_approx);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Cce.ComputeSecondOrderRadialDerivativeJ",
    "[Unit][Cce]") {
  for (const size_t l_max : {8_st, 10_st}) {
    test_h_equation_is_satisfied(l_max);
  }
}
}  // namespace
}  // namespace Cce::InitializeJ::CauchySecondOrder_detail
