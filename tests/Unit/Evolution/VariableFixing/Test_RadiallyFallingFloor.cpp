// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// ThermoDim = 1
void test_eos_call(
    const VariableFixing::RadiallyFallingFloor<3>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  Scalar<DataVector> initial_pressure{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}};
  Scalar<DataVector> pressure = initial_pressure;
  Scalar<DataVector> initial_density{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}};
  Scalar<DataVector> density = initial_density;
  Scalar<DataVector> initial_specific_internal_energy{
      DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> specific_internal_energy =
      initial_specific_internal_energy;
  Scalar<DataVector> initial_temperature{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> temperature = initial_temperature;
  Scalar<DataVector> initial_electron_fraction{
      DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> electron_fraction = initial_electron_fraction;
  Scalar<DataVector> initial_specific_enthalpy{
      DataVector{1.0, 1.0, 1.0, 1.0, 1.0}};
  Scalar<DataVector> specific_enthalpy = initial_specific_enthalpy;

  for (size_t i = 0; i < initial_specific_enthalpy.size(); i++) {
    initial_specific_enthalpy[i] += initial_specific_internal_energy[i] +
                                    initial_pressure[i] / initial_density[i];

    specific_enthalpy[i] +=
        specific_internal_energy[i] + pressure[i] / density[i];
  }

  variable_fixer(&density, &pressure, &specific_internal_energy,
                 &specific_enthalpy, &temperature, &electron_fraction, coords,
                 equation_of_state);

  // Density should change
  CHECK_FALSE(initial_density == density);

  // Pressure should change
  CHECK_FALSE(initial_pressure == pressure);

  // Specific internal energy should change
  CHECK_FALSE(initial_specific_internal_energy == specific_internal_energy);

  // specific enthalpy should change
  CHECK_FALSE(initial_specific_enthalpy == specific_enthalpy);

  // temperature should change (for 1D EoS)
  CHECK_FALSE(initial_temperature == temperature);

  // Ye should remain unchanged (for 1D EoS)
  CHECK_ITERABLE_APPROX(initial_electron_fraction, electron_fraction);
}

// ThermoDim = 2
void test_eos_call(
    const VariableFixing::RadiallyFallingFloor<3>& variable_fixer,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  Scalar<DataVector> initial_pressure{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}};
  Scalar<DataVector> pressure = initial_pressure;
  Scalar<DataVector> initial_density{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}};
  Scalar<DataVector> density = initial_density;
  Scalar<DataVector> initial_specific_internal_energy{
      DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> specific_internal_energy =
      initial_specific_internal_energy;
  Scalar<DataVector> initial_temperature{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> temperature = initial_temperature;
  Scalar<DataVector> initial_electron_fraction{
      DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> electron_fraction = initial_electron_fraction;
  Scalar<DataVector> initial_specific_enthalpy{
      DataVector{1.0, 1.0, 1.0, 1.0, 1.0}};
  Scalar<DataVector> specific_enthalpy = initial_specific_enthalpy;

  for (size_t i = 0; i < initial_specific_enthalpy.size(); i++) {
    initial_specific_enthalpy[i] += initial_specific_internal_energy[i] +
                                    initial_pressure[i] / initial_density[i];

    specific_enthalpy[i] +=
        specific_internal_energy[i] + pressure[i] / density[i];
  }

  variable_fixer(&density, &pressure, &specific_internal_energy,
                 &specific_enthalpy, &temperature, &electron_fraction, coords,
                 equation_of_state);

  // Density should change
  CHECK_FALSE(initial_density == density);

  // Pressure should change
  CHECK_FALSE(initial_pressure == pressure);

  // Specific internal energy should change
  CHECK_FALSE(initial_specific_internal_energy == specific_internal_energy);

  // specific enthalpy should change
  CHECK_FALSE(initial_specific_enthalpy == specific_enthalpy);

  // temperature should change (for 2D EoS)
  CHECK_FALSE(initial_temperature == temperature);

  // Ye should remain unchanged (for 2D EoS)
  CHECK_ITERABLE_APPROX(initial_electron_fraction, electron_fraction);
}

void test_variable_fixer(
    const VariableFixing::RadiallyFallingFloor<3>& variable_fixer) {
  Scalar<DataVector> pressure{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}};
  Scalar<DataVector> density{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}};
  Scalar<DataVector> specific_internal_energy{
      DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> temperature{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> electron_fraction{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}};
  Scalar<DataVector> specific_enthalpy{DataVector{1.0, 1.0, 1.0, 1.0, 1.0}};

  for (size_t i = 0; i < specific_enthalpy.get().size(); i++) {
    specific_enthalpy.get()[i] += specific_internal_energy.get()[i] +
                                  pressure.get()[i] / density.get()[i];
  }

  const double root_three = sqrt(3.0);
  const DataVector expected_density{
      2.3, 1.e-5 * pow(3, -0.75),
      1.e-10,  // quantities at a radius below
               // `radius_at_which_to_begin_applying_floor` do not get fixed.
      1.e-5 * pow(3, -0.75), 1.e-5 * pow(2.0 * root_three, -1.5)};
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};
  tnsr::I<DataVector, 3, Frame::Inertial> coords{{{x, y, z}}};

  const EquationsOfState::PolytropicFluid<true> polytrope{1.0, 2.0};
  const EquationsOfState::IdealFluid<true> ideal_fluid{5.0 / 3.0};

  variable_fixer(&density, &pressure, &specific_internal_energy,
                 &specific_enthalpy, &temperature, &electron_fraction, coords,
                 polytrope);

  auto expected_pressure = get(polytrope.pressure_from_density(density));
  // The i = 2 entry should not change, b/c the radial coordinate <
  // minimum_radius_at_which_to_apply_floor_
  expected_pressure[2] = pressure.get()[2];

  CHECK_ITERABLE_APPROX(pressure.get(), expected_pressure);

  CHECK_ITERABLE_APPROX(density.get(), expected_density);

  // Ensure eos calls change proper values

  // 1D
  test_eos_call(variable_fixer, polytrope, coords);

  // 2D
  test_eos_call(variable_fixer, ideal_fluid, coords);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.RadiallyFallingFloor",
                  "[VariableFixing][Unit]") {
  VariableFixing::RadiallyFallingFloor<3> variable_fixer{1.e-4, 1.e-5, -1.5,
                                                         1.e-7 / 3.0, -2.5};
  test_variable_fixer(variable_fixer);
  test_serialization(variable_fixer);

  const auto fixer_from_options =
      TestHelpers::test_creation<VariableFixing::RadiallyFallingFloor<3>>(
          "MinimumRadius: 1.e-4\n"
          "ScaleDensityFloor: 1.e-5\n"
          "PowerDensityFloor: -1.5\n"
          "ScalePressureFloor: 0.33333333333333333e-7\n"
          "PowerPressureFloor: -2.5\n");
  test_variable_fixer(fixer_from_options);
}
