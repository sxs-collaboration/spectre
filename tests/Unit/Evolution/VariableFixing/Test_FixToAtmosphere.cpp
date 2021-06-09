// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare VariableFixing::FixToAtmosphere

namespace {

template <size_t Dim>
void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<Dim>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 1>&
        equation_of_state) noexcept {
  // 2.e-12 -> below cutoff, velocity goes to zero
  // 2.e-11 -> above cutoff, no changes
  // 4.e-12 -> density unchanged, velocity restricted
  Scalar<DataVector> density{DataVector{2.e-12, 2.e-11, 4.e-12}};
  auto pressure = equation_of_state.pressure_from_density(density);
  auto specific_enthalpy =
      equation_of_state.specific_enthalpy_from_density(density);
  auto specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(density);

  Scalar<DataVector> lorentz_factor{
      DataVector{5.0 / 3.0, 1.25, 1.8898223650461359}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  spatial_velocity.get(0) = DataVector{0.8, 0.6, 0.6};
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    spatial_metric.get(i, i) = 2.0;
  }
  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &specific_enthalpy, spatial_metric,
                 equation_of_state);

  Scalar<DataVector> expected_density{DataVector{1.e-12, 2.e-11, 4.e-12}};
  auto expected_pressure =
      equation_of_state.pressure_from_density(expected_density);
  auto expected_specific_enthalpy =
      equation_of_state.specific_enthalpy_from_density(expected_density);
  auto expected_specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(expected_density);
  Scalar<DataVector> expected_lorentz_factor{
      DataVector{1.0, 1.25, 1.0000000001020408}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  expected_spatial_velocity.get(0)[1] = 0.6;
  // The [2] component of the expected velocity is:
  //     velocity *= max_velocity_magnitude_ * (rho - rho_cut) /
  //                 (trans_rho - rho_cut) / |v^i|
  expected_spatial_velocity.get(0)[2] = 0.6 * (4.e-12 - 3.e-12) /
                                        (1.e-11 - 3.e-12) * 1.e-4 /
                                        sqrt(0.6 * 0.6 * 2.0);

  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(specific_enthalpy, expected_specific_enthalpy);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);
  CHECK_ITERABLE_APPROX(lorentz_factor, expected_lorentz_factor);
  CHECK_ITERABLE_APPROX(spatial_velocity, expected_spatial_velocity);
}

template <size_t Dim>
void test_variable_fixer(
    const VariableFixing::FixToAtmosphere<Dim>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state) {
  Scalar<DataVector> density{DataVector{2.e-12, 2.e-11, 4.e-12}};
  Scalar<DataVector> specific_internal_energy{DataVector{2.0, 3.0, 3.0}};
  auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);
  auto specific_enthalpy =
      equation_of_state.specific_enthalpy_from_density_and_energy(
          density, specific_internal_energy);

  Scalar<DataVector> lorentz_factor{
      DataVector{5.0 / 3.0, 1.25, 1.8898223650461359}};
  auto spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  spatial_velocity.get(0) = DataVector{0.8, 0.6, 0.6};
  auto spatial_metric =
      make_with_value<tnsr::ii<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    spatial_metric.get(i, i) = 2.0;
  }
  variable_fixer(&density, &specific_internal_energy, &spatial_velocity,
                 &lorentz_factor, &pressure, &specific_enthalpy, spatial_metric,
                 equation_of_state);

  Scalar<DataVector> expected_density{DataVector{1.e-12, 2.e-11, 4.e-12}};
  Scalar<DataVector> expected_specific_internal_energy{
      DataVector{0.0, 3.0, 3.0}};
  auto expected_pressure = equation_of_state.pressure_from_density_and_energy(
      expected_density, expected_specific_internal_energy);
  auto expected_specific_enthalpy =
      equation_of_state.specific_enthalpy_from_density_and_energy(
          expected_density, expected_specific_internal_energy);
  Scalar<DataVector> expected_lorentz_factor{
      DataVector{1.0, 1.25, 1.0000000001020408}};
  auto expected_spatial_velocity =
      make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(density, 0.0);
  expected_spatial_velocity.get(0)[1] = 0.6;
  // The [2] component of the expected velocity is:
  //     velocity *= max_velocity_magnitude_ * (rho - rho_cut) /
  //                 (trans_rho - rho_cut) / |v^i|
  expected_spatial_velocity.get(0)[2] = 0.6 * (4.e-12 - 3.e-12) /
                                        (1.e-11 - 3.e-12) * 1.e-4 /
                                        sqrt(0.6 * 0.6 * 2.0);

  CHECK_ITERABLE_APPROX(density, expected_density);
  CHECK_ITERABLE_APPROX(pressure, expected_pressure);
  CHECK_ITERABLE_APPROX(specific_enthalpy, expected_specific_enthalpy);
  CHECK_ITERABLE_APPROX(specific_internal_energy,
                        expected_specific_internal_energy);
  CHECK_ITERABLE_APPROX(lorentz_factor, expected_lorentz_factor);
  CHECK_ITERABLE_APPROX(spatial_velocity, expected_spatial_velocity);
}

template <size_t Dim>
void test_variable_fixer() noexcept {
  // Test for representative 1-d equation of state
  VariableFixing::FixToAtmosphere<Dim> variable_fixer{1.e-12, 3.e-12, 1.e-11,
                                                      1.e-4};
  EquationsOfState::PolytropicFluid<true> polytrope{1.0, 2.0};
  test_variable_fixer<Dim>(variable_fixer, polytrope);
  test_serialization(variable_fixer);

  const auto fixer_from_options =
      TestHelpers::test_creation<VariableFixing::FixToAtmosphere<Dim>>(
          "DensityOfAtmosphere: 1.0e-12\n"
          "DensityCutoff: 3.0e-12\n"
          "TransitionDensityCutoff: 1.0e-11\n"
          "MaxVelocityMagnitude: 1.0e-4\n");
  test_variable_fixer(fixer_from_options, polytrope);

  // Test for representative 2-d equation of state
  EquationsOfState::IdealFluid<true> ideal_fluid{5.0 / 3.0};
  test_variable_fixer<Dim>(variable_fixer, ideal_fluid);
  test_serialization(variable_fixer);

  test_variable_fixer<Dim>(fixer_from_options, ideal_fluid);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.FixToAtmosphere",
                  "[VariableFixing][Unit]") {
  test_variable_fixer<1>();
  test_variable_fixer<2>();
  test_variable_fixer<3>();
}
