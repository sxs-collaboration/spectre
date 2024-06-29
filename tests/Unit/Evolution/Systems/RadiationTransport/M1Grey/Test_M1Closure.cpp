// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"  // IWYU pragma: keep
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
tnsr::a<DataVector, 3> add_time_component_from_normal(
    const tnsr::i<DataVector, 3>& three_covector,
    const tnsr::A<DataVector, 3>& normal_vector,
    const Scalar<DataVector>& vector_dot_normal) {
  auto four_covector =
      make_with_value<tnsr::a<DataVector, 3>>(three_covector, 0.0);
  const auto temporal_part = tenex::evaluate(
      vector_dot_normal() - three_covector(ti::i) * normal_vector(ti::I));
  tenex::evaluate<ti::t>(make_not_null(&four_covector),
                         temporal_part() / normal_vector(ti::T));
  tenex::evaluate<ti::i>(make_not_null(&four_covector), three_covector(ti::i));

  INFO("Test-internal 4-vector creation");
  CHECK_ITERABLE_APPROX(
      tenex::evaluate(four_covector(ti::a) * normal_vector(ti::A)),
      vector_dot_normal);

  return four_covector;
}

void check_closure_consistency(
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::I<DataVector, 3>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const Scalar<DataVector>& energy_density,
    const tnsr::i<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& closure_factor,
    const Scalar<DataVector>& comoving_energy_density,
    const tnsr::i<DataVector, 3>& comoving_momentum_density_spatial,
    const Scalar<DataVector>& comoving_momentum_density_normal,
    const tnsr::II<DataVector, 3>& pressure_tensor) {
  // The closure calculation does a root find to 6 digits.
  auto closure_approx = Approx::custom().epsilon(1.0e-5).scale(1.0);

  const auto zero = make_with_value<Scalar<DataVector>>(spatial_metric, 0.0);
  // Lapse and shift don't come into the closure equations, so set
  // them to one and zero for simplicity.
  auto spacetime_metric = make_with_value<tnsr::aa<DataVector, 3>>(zero, 0.0);
  tenex::evaluate<ti::t, ti::t>(make_not_null(&spacetime_metric), -1.0);
  tenex::evaluate<ti::i, ti::j>(make_not_null(&spacetime_metric),
                                spatial_metric(ti::i, ti::j));
  const auto inverse_spacetime_metric =
      determinant_and_inverse(spacetime_metric).second;

  auto normal_covector = make_with_value<tnsr::a<DataVector, 3>>(zero, 0.0);
  tenex::evaluate<ti::t>(make_not_null(&normal_covector), -1.0);
  const auto normal_vector = tenex::evaluate<ti::A>(
      inverse_spacetime_metric(ti::A, ti::B) * normal_covector(ti::b));

  auto fluid_four_velocity = make_with_value<tnsr::A<DataVector, 3>>(zero, 0.0);
  {
    tenex::evaluate<ti::I>(make_not_null(&fluid_four_velocity),
                           fluid_lorentz_factor() * fluid_velocity(ti::I));
    const auto three_norm = tenex::evaluate(spacetime_metric(ti::a, ti::b) *
                                            fluid_four_velocity(ti::A) *
                                            fluid_four_velocity(ti::B));
    // Shift = 0
    tenex::evaluate<ti::T>(
        make_not_null(&fluid_four_velocity),
        sqrt(-(three_norm() + 1.0) / spacetime_metric(ti::t, ti::t)));
  }

  const auto momentum_density4 =
      add_time_component_from_normal(momentum_density, normal_vector, zero);
  const auto comoving_momentum_density4 = add_time_component_from_normal(
      comoving_momentum_density_spatial, normal_vector,
      comoving_momentum_density_normal);

  // Check comoving momentum density is spatial in the fluid frame
  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate(comoving_momentum_density4(ti::a) *
                      fluid_four_velocity(ti::A)),
      zero, closure_approx);

  // Consistency of stress-energy tensor in two frames
  const auto stress_energy_from_eulerian = tenex::evaluate<ti::a, ti::b>(
      energy_density() * normal_covector(ti::a) * normal_covector(ti::b) +
      momentum_density4(ti::a) * normal_covector(ti::b) +
      normal_covector(ti::a) * momentum_density4(ti::b) +
      spacetime_metric(ti::a, ti::i) * spacetime_metric(ti::b, ti::j) *
          pressure_tensor(ti::I, ti::J));
  // We don't have the fluid-frame pressure, so this is missing that
  // contribution.  We will only check a projection that this pressure
  // does not contribute to.
  const auto stress_energy_from_comoving = tenex::evaluate<ti::a, ti::b>(
      comoving_energy_density() * spacetime_metric(ti::a, ti::c) *
          fluid_four_velocity(ti::C) * spacetime_metric(ti::b, ti::d) *
          fluid_four_velocity(ti::D) +
      comoving_momentum_density4(ti::a) * spacetime_metric(ti::b, ti::d) *
          fluid_four_velocity(ti::D) +
      spacetime_metric(ti::a, ti::c) * fluid_four_velocity(ti::C) *
          comoving_momentum_density4(ti::b));
  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate<ti::a>(stress_energy_from_eulerian(ti::a, ti::b) *
                             fluid_four_velocity(ti::B)),
      tenex::evaluate<ti::a>(stress_energy_from_comoving(ti::a, ti::b) *
                             fluid_four_velocity(ti::B)),
      closure_approx);

  // Check the choice of the closure factor
  CHECK_ITERABLE_CUSTOM_APPROX(
      tenex::evaluate(square(closure_factor()) *
                      square(comoving_energy_density())),
      tenex::evaluate(comoving_momentum_density4(ti::a) *
                      inverse_spacetime_metric(ti::A, ti::B) *
                      comoving_momentum_density4(ti::b)),
      closure_approx);
}

void check_limits() {
  const size_t used_for_size = 5;
  RadiationTransport::M1Grey::ComputeM1Closure<
      tmpl::list<neutrinos::ElectronNeutrinos<0>>>
      closure;
  // Create variables
  // Input
  Scalar<DataVector> energy_density(used_for_size);
  tnsr::i<DataVector, 3, Frame::Inertial> momentum_density(used_for_size);
  tnsr::I<DataVector, 3, Frame::Inertial> fluid_velocity(used_for_size);
  Scalar<DataVector> fluid_lorentz_factor(used_for_size);
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric(used_for_size);
  // Output
  Scalar<DataVector> closure_factor(used_for_size);
  tnsr::II<DataVector, 3, Frame::Inertial> pressure_tensor(used_for_size);
  Scalar<DataVector> comoving_energy_density(used_for_size);
  Scalar<DataVector> comoving_momentum_density_normal(used_for_size);
  tnsr::i<DataVector, 3, Frame::Inertial> comoving_momentum_density_spatial(
      used_for_size);

  // Accuracy required for closure factor
  static Approx custom_approx = Approx::custom().epsilon(1.e-5).scale(1.0);

  // Set fluid/metric variables
  for (size_t m = 0; m < 3; m++) {
    fluid_velocity.get(m) = 0.1 * m;
    spatial_metric.get(m, m) = 1. + 0.1 * m * m;
    for (size_t n = m + 1; n < 3; n++) {
      spatial_metric.get(m, n) = 0.1 * (m + n);
    }
  }
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  get(fluid_lorentz_factor) =
      1. / sqrt(1. - get(dot_product(fluid_velocity, fluid_velocity,
                                     spatial_metric)));

  // Initialize closure factor (as the input value is used as initial
  // guess for the root finding algorithm).
  get(closure_factor) = -1.;
  // (1) Optically thick limit:
  Scalar<DataVector> four_thirds_square_fluid_lorentz_factor(
      4. / 3. * square(get(fluid_lorentz_factor)));
  get(energy_density) = get(four_thirds_square_fluid_lorentz_factor) - 1. / 3.;
  for (int m = 0; m < 3; m++) {
    momentum_density.get(m) = 0.;
    for (int n = 0; n < 3; n++) {
      momentum_density.get(m) += get(four_thirds_square_fluid_lorentz_factor) *
                                 fluid_velocity.get(n) *
                                 spatial_metric.get(m, n);
    }
  }
  closure.apply(make_not_null(&closure_factor), make_not_null(&pressure_tensor),
                make_not_null(&comoving_energy_density),
                make_not_null(&comoving_momentum_density_normal),
                make_not_null(&comoving_momentum_density_spatial),
                energy_density, momentum_density, fluid_velocity,
                fluid_lorentz_factor, spatial_metric, inv_spatial_metric);
  const DataVector expected_xi0{0.0, 0.0, 0.0, 0.0, 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(get(closure_factor), expected_xi0,
                               custom_approx);
  check_closure_consistency(
      spatial_metric, fluid_velocity, fluid_lorentz_factor, energy_density,
      momentum_density, closure_factor, comoving_energy_density,
      comoving_momentum_density_spatial, comoving_momentum_density_normal,
      pressure_tensor);

  // (2) Optically thin limit
  momentum_density.get(0) = -1.;
  momentum_density.get(1) = 5.;
  momentum_density.get(2) = 3.;
  energy_density = magnitude(momentum_density, inv_spatial_metric);
  closure.apply(make_not_null(&closure_factor), make_not_null(&pressure_tensor),
                make_not_null(&comoving_energy_density),
                make_not_null(&comoving_momentum_density_normal),
                make_not_null(&comoving_momentum_density_spatial),
                energy_density, momentum_density, fluid_velocity,
                fluid_lorentz_factor, spatial_metric, inv_spatial_metric);
  const DataVector expected_xi1{1.0, 1.0, 1.0, 1.0, 1.0};
  CHECK_ITERABLE_CUSTOM_APPROX(get(closure_factor), expected_xi1,
                               custom_approx);
  check_closure_consistency(
      spatial_metric, fluid_velocity, fluid_lorentz_factor, energy_density,
      momentum_density, closure_factor, comoving_energy_density,
      comoving_momentum_density_spatial, comoving_momentum_density_normal,
      pressure_tensor);
}

void check_random() {
  const size_t used_for_size = 5;
  using closure = RadiationTransport::M1Grey::ComputeM1Closure<
      tmpl::list<neutrinos::ElectronNeutrinos<0>>>;

  MAKE_GENERATOR(gen);

  std::uniform_real_distribution metric_distribution(-0.1, 0.1);
  std::uniform_real_distribution momentum_distribution(-10.0, 10.0);
  std::uniform_real_distribution velocity_magnitude_distribution(0.0, 0.9);

  auto spatial_metric =
      make_with_random_values<tnsr::ii<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&metric_distribution),
          used_for_size);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) += 1.0;
  }
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto momentum_density =
      make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
          make_not_null(&gen), make_not_null(&momentum_distribution),
          used_for_size);
  // We can take shift = 0, and then as momentum_magnitude^0 = 0 the
  // spacetime norm and the spatial norm are the same.
  const auto momentum_magnitude =
      magnitude(momentum_density, inv_spatial_metric);
  Scalar<DataVector> energy_density(used_for_size);
  for (size_t i = 0; i < get(energy_density).size(); ++i) {
    std::uniform_real_distribution energy_distribution(
        get(momentum_magnitude)[i], 2.0 * get(momentum_magnitude)[i]);
    get(energy_density)[i] = energy_distribution(gen);
  }

  // Test with zero fluid velocity (special case in calculation)
  auto fluid_velocity =
      make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.0);
  auto fluid_lorentz_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);

  // Output
  Scalar<DataVector> closure_factor(used_for_size);
  tnsr::II<DataVector, 3, Frame::Inertial> pressure_tensor(used_for_size);
  Scalar<DataVector> comoving_energy_density(used_for_size);
  Scalar<DataVector> comoving_momentum_density_normal(used_for_size);
  tnsr::i<DataVector, 3, Frame::Inertial> comoving_momentum_density_spatial(
      used_for_size);

  // Initialize closure factor (as the input value is used as initial
  // guess for the root finding algorithm).
  get(closure_factor) = -1.;

  closure::apply(make_not_null(&closure_factor),
                 make_not_null(&pressure_tensor),
                 make_not_null(&comoving_energy_density),
                 make_not_null(&comoving_momentum_density_normal),
                 make_not_null(&comoving_momentum_density_spatial),
                 energy_density, momentum_density, fluid_velocity,
                 fluid_lorentz_factor, spatial_metric, inv_spatial_metric);
  check_closure_consistency(
      spatial_metric, fluid_velocity, fluid_lorentz_factor, energy_density,
      momentum_density, closure_factor, comoving_energy_density,
      comoving_momentum_density_spatial, comoving_momentum_density_normal,
      pressure_tensor);
  // The two frames are the same
  CHECK(energy_density == comoving_energy_density);
  CHECK(momentum_density == comoving_momentum_density_spatial);
  // Test continuity near the zero
  {
    fluid_velocity.get(0) = 1.0e-6;
    tenex::evaluate(
        make_not_null(&fluid_lorentz_factor),
        1.0 / sqrt(1.0 - fluid_velocity(ti::I) * spatial_metric(ti::i, ti::j) *
                             fluid_velocity(ti::J)));
    Approx continuity_approx = Approx::custom().epsilon(1.0e-4).scale(1.0);
    Scalar<DataVector> closure_factor2(used_for_size);
    tnsr::II<DataVector, 3, Frame::Inertial> pressure_tensor2(used_for_size);
    Scalar<DataVector> comoving_energy_density2(used_for_size);
    Scalar<DataVector> comoving_momentum_density_normal2(used_for_size);
    tnsr::i<DataVector, 3, Frame::Inertial> comoving_momentum_density_spatial2(
        used_for_size);
    get(closure_factor2) = -1.;
    closure::apply(make_not_null(&closure_factor2),
                   make_not_null(&pressure_tensor2),
                   make_not_null(&comoving_energy_density2),
                   make_not_null(&comoving_momentum_density_normal2),
                   make_not_null(&comoving_momentum_density_spatial2),
                   energy_density, momentum_density, fluid_velocity,
                   fluid_lorentz_factor, spatial_metric, inv_spatial_metric);
    CHECK_ITERABLE_CUSTOM_APPROX(closure_factor, closure_factor2,
                                 continuity_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(pressure_tensor, pressure_tensor2,
                                 continuity_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(comoving_energy_density,
                                 comoving_energy_density2, continuity_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(comoving_momentum_density_normal,
                                 comoving_momentum_density_normal2,
                                 continuity_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(comoving_momentum_density_spatial,
                                 comoving_momentum_density_spatial2,
                                 continuity_approx);
  }

  // Test with a random fluid velocity.  With the random metric, it is
  // hard to generate a valid velocity in one step, so we choose the
  // direction and magnitude separately.
  fill_with_random_values(make_not_null(&fluid_velocity), make_not_null(&gen),
                          make_not_null(&momentum_distribution));
  const auto velocity_magnitude = make_with_random_values<Scalar<DataVector>>(
      make_not_null(&gen), make_not_null(&velocity_magnitude_distribution),
      used_for_size);
  {
    const DataVector velocity_scale =
        get(velocity_magnitude) /
        get(magnitude(fluid_velocity, spatial_metric));
    for (size_t i = 0; i < 3; ++i) {
      fluid_velocity.get(i) *= velocity_scale;
    }
  }

  get(fluid_lorentz_factor) = 1.0 / sqrt(1.0 - square(get(velocity_magnitude)));

  closure::apply(make_not_null(&closure_factor),
                 make_not_null(&pressure_tensor),
                 make_not_null(&comoving_energy_density),
                 make_not_null(&comoving_momentum_density_normal),
                 make_not_null(&comoving_momentum_density_spatial),
                 energy_density, momentum_density, fluid_velocity,
                 fluid_lorentz_factor, spatial_metric, inv_spatial_metric);
  check_closure_consistency(
      spatial_metric, fluid_velocity, fluid_lorentz_factor, energy_density,
      momentum_density, closure_factor, comoving_energy_density,
      comoving_momentum_density_spatial, comoving_momentum_density_normal,
      pressure_tensor);
}
}  // namespace

SPECTRE_TEST_CASE("Evolution.Systems.RadiationTransport.M1Grey.M1Closure",
                  "[Unit][M1Grey]") {
  check_limits();
  check_random();
}
