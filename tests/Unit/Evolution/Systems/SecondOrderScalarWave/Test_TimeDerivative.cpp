// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// The volume time derivatives are the exact, linear relations
// \f$\partial_t \Psi = -\Pi\f$ and
// \f$\partial_t \Pi = -\delta^{ij}\partial_i \Phi_j\f$, so we can check them
// against arbitrary random inputs without invoking an analytic solution.
template <size_t Dim>
void test_time_derivative(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto nn_distribution = make_not_null(&distribution);
  const size_t num_pts = 5;
  const DataVector used_for_size(num_pts);

  const auto pi = make_with_random_values<Scalar<DataVector>>(
      generator, nn_distribution, used_for_size);
  // The evolved variables' derivatives are unused by the time derivative
  // (they are in gradient_variables only for the framework's moving-mesh
  // term); random values prove they do not enter the result.
  const auto d_psi =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          generator, nn_distribution, used_for_size);
  const auto d_pi =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          generator, nn_distribution, used_for_size);
  const auto d_phi =
      make_with_random_values<tnsr::ij<DataVector, Dim, Frame::Inertial>>(
          generator, nn_distribution, used_for_size);

  Scalar<DataVector> dt_psi{num_pts};
  Scalar<DataVector> dt_pi{num_pts};
  const auto decisions = SecondOrderScalarWave::TimeDerivative<Dim>::apply(
      make_not_null(&dt_psi), make_not_null(&dt_pi), d_psi, d_pi, d_phi, pi);

  CHECK_ITERABLE_APPROX(dt_psi, Scalar<DataVector>(-1.0 * get(pi)));

  DataVector expected_dt_pi(num_pts, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    expected_dt_pi -= d_phi.get(d, d);
  }
  CHECK_ITERABLE_APPROX(dt_pi, Scalar<DataVector>(expected_dt_pi));

  // The LDG scheme is flux-free, so no flux divergence is computed.
  CHECK(decisions.compute_flux_divergence == false);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SecondOrderScalarWave.TimeDerivative",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  test_time_derivative<1>(make_not_null(&generator));
  test_time_derivative<2>(make_not_null(&generator));
  test_time_derivative<3>(make_not_null(&generator));
}
