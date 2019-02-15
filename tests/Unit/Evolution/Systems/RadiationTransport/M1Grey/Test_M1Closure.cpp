// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

/// Test M1 closure function
SPECTRE_TEST_CASE("Evolution.Systems.RadiationTransport.M1Grey.M1Closure",
                  "[Unit][M1Grey]") {
  const DataVector used_for_size(5);
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
  RadiationTransport::M1Grey::M1Closure(
      make_not_null(&closure_factor), make_not_null(&pressure_tensor),
      make_not_null(&comoving_energy_density),
      make_not_null(&comoving_momentum_density_normal),
      make_not_null(&comoving_momentum_density_spatial), energy_density,
      momentum_density, fluid_velocity, fluid_lorentz_factor, spatial_metric,
      inv_spatial_metric);
  CHECK((get(closure_factor)[0] == approx(0.).epsilon(1.e-5)));

  // (2) Optically thin limit
  momentum_density.get(0) = -1.;
  momentum_density.get(1) = 5.;
  momentum_density.get(2) = 3.;
  energy_density = magnitude(momentum_density, inv_spatial_metric);
  RadiationTransport::M1Grey::M1Closure(
      make_not_null(&closure_factor), make_not_null(&pressure_tensor),
      make_not_null(&comoving_energy_density),
      make_not_null(&comoving_momentum_density_normal),
      make_not_null(&comoving_momentum_density_spatial), energy_density,
      momentum_density, fluid_velocity, fluid_lorentz_factor, spatial_metric,
      inv_spatial_metric);
  CHECK((get(closure_factor)[0] == approx(1.).epsilon(1.e-5)));
}
