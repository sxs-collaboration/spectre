// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare VariableFixing::FixToAtmosphere

namespace {

void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<1>& variable_fixer) {
  EquationsOfState::PolytropicFluid<true> polytrope{1.0, 2.0};

  Scalar<DataVector> density{DataVector{2.e-12, 2.e-11}};
  auto pressure = polytrope.pressure_from_density(density);
  auto specific_enthalpy = polytrope.specific_enthalpy_from_density(density);
  auto specific_internal_energy =
      polytrope.specific_internal_energy_from_density(density);

  Scalar<DataVector> lorentz_factor{DataVector{5.0 / 3.0, 1.25}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(density, 0.0);
  spatial_velocity.get(0) = DataVector{0.8, 0.6};
  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &specific_enthalpy, polytrope);

  Scalar<DataVector> expected_density{DataVector{1.e-12, 2.e-11}};
  auto expected_pressure = polytrope.pressure_from_density(expected_density);
  auto expected_specific_enthalpy =
      polytrope.specific_enthalpy_from_density(expected_density);
  auto expected_specific_internal_energy =
      polytrope.specific_internal_energy_from_density(expected_density);
  Scalar<DataVector> expected_lorentz_factor{DataVector{1.0, 1.25}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(density, 0.0);
  expected_spatial_velocity.get(0)[1] = 0.6;

  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(specific_enthalpy, expected_specific_enthalpy);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);
  CHECK_ITERABLE_APPROX(lorentz_factor, expected_lorentz_factor);
  CHECK_ITERABLE_APPROX(spatial_velocity, expected_spatial_velocity);
}

void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<2>& variable_fixer) {
  EquationsOfState::IdealFluid<true> ideal_fluid{5.0 / 3.0};

  Scalar<DataVector> density{DataVector{2.e-12, 2.e-11}};
  Scalar<DataVector> specific_internal_energy{DataVector{2.0, 3.0}};
  auto pressure = ideal_fluid.pressure_from_density_and_energy(
      density, specific_internal_energy);
  auto specific_enthalpy =
      ideal_fluid.specific_enthalpy_from_density_and_energy(
          density, specific_internal_energy);

  Scalar<DataVector> lorentz_factor{DataVector{5.0 / 3.0, 1.25}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(density, 0.0);
  spatial_velocity.get(0) = DataVector{0.8, 0.6};
  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &specific_enthalpy, ideal_fluid);

  Scalar<DataVector> expected_density{DataVector{1.e-12, 2.e-11}};
  Scalar<DataVector> expected_specific_internal_energy{DataVector{0.0, 3.0}};
  auto expected_pressure = ideal_fluid.pressure_from_density_and_energy(
      expected_density, expected_specific_internal_energy);
  auto expected_specific_enthalpy =
      ideal_fluid.specific_enthalpy_from_density_and_energy(
          expected_density, expected_specific_internal_energy);
  Scalar<DataVector> expected_lorentz_factor{DataVector{1.0, 1.25}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(density, 0.0);
  expected_spatial_velocity.get(0)[1] = 0.6;

  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(specific_enthalpy, expected_specific_enthalpy);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);
  CHECK_ITERABLE_APPROX(lorentz_factor, expected_lorentz_factor);
  CHECK_ITERABLE_APPROX(spatial_velocity, expected_spatial_velocity);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.FixToAtmosphere",
                  "[VariableFixing][Unit]") {
  VariableFixing::FixToAtmosphere<1> variable_fixer_1d{1.e-12, 1.e-11};
  test_variable_fixer(variable_fixer_1d);
  test_serialization(variable_fixer_1d);

  const auto fixer_from_options_1d =
      test_creation<VariableFixing::FixToAtmosphere<1>>(
          "  DensityOfAtmosphere: 1.0e-12\n"
          "  DensityCutoff: 1.0e-11\n");
  test_variable_fixer(fixer_from_options_1d);

  VariableFixing::FixToAtmosphere<2> variable_fixer_2d{1.e-12, 1.e-11};
  test_variable_fixer(variable_fixer_2d);
  test_serialization(variable_fixer_2d);

  const auto fixer_from_options_2d =
      test_creation<VariableFixing::FixToAtmosphere<2>>(
          "  DensityOfAtmosphere: 1.0e-12\n"
          "  DensityCutoff: 1.0e-11\n");
  test_variable_fixer(fixer_from_options_2d);
}
