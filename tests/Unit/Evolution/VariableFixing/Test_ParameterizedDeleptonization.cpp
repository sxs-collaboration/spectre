// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/VariableFixing/ParameterizedDeleptonization.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_variable_fixer(
    const VariableFixing::ParameterizedDeleptonization& variable_fixer,
    const EquationsOfState::EquationOfState<true, 1>& equation_of_state) {
  // 8.0e-11 -> below cutoff (less dense/further out in the star), picks high Ye
  // 1.5e-10 -> above cutoff (high dense/closer to star center), pick low Ye
  // 3.0e-10 -> density transition regions, should be in between

  const Scalar<DataVector> density{DataVector{0.8e-10, 1.5e-10, 3e-10}};
  Scalar<DataVector> electron_fraction{DataVector{0.5, 0.5, 0.5}};
  // These should remain unchanged, sinced param delep can only make Ye drop
  Scalar<DataVector> electron_fraction_low{DataVector{0.1, 0.15, 0.2}};

  auto pressure = equation_of_state.pressure_from_density(density);
  auto specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density(density);
  auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      density, specific_internal_energy, pressure);

  // check Ye values that will naturally decrease
  variable_fixer(&specific_internal_energy, &electron_fraction, &pressure,
                 &specific_enthalpy, density, equation_of_state);

  const Scalar<DataVector> expected_electron_fraction{
      DataVector{0.5, 0.40980370118030457, 0.285}};

  CHECK_ITERABLE_APPROX(electron_fraction, expected_electron_fraction);

  // check Ye values that will stay constant
  const Scalar<DataVector> expected_electron_fraction_low{
      electron_fraction_low};

  variable_fixer(&specific_internal_energy, &electron_fraction_low, &pressure,
                 &specific_enthalpy, density, equation_of_state);

  CHECK_ITERABLE_APPROX(electron_fraction_low, expected_electron_fraction_low);
}

// 2D EoS version
void test_variable_fixer(
    const VariableFixing::ParameterizedDeleptonization& variable_fixer,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state) {
  const Scalar<DataVector> density{DataVector{3.0e-9, 3.7e-9, 4.0e-9}};
  Scalar<DataVector> electron_fraction{DataVector{0.5, 0.5, 0.5}};
  // These should remain unchanged, sinced param delep can only make Ye drop
  Scalar<DataVector> electron_fraction_low{DataVector{0.1, 0.15, 0.2}};

  Scalar<DataVector> specific_internal_energy{DataVector{1.0, 2.0, 3.0}};
  auto pressure = equation_of_state.pressure_from_density_and_energy(
      density, specific_internal_energy);
  auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      density, specific_internal_energy, pressure);

  // check Ye values that will naturally decrease
  variable_fixer(&specific_internal_energy, &electron_fraction, &pressure,
                 &specific_enthalpy, density, equation_of_state);

  const Scalar<DataVector> expected_electron_fraction{
      DataVector{0.49, 0.3077746733472282, 0.2}};

  CHECK_ITERABLE_APPROX(electron_fraction, expected_electron_fraction);

  // check Ye values that will stay constant
  const Scalar<DataVector> expected_electron_fraction_low{
      electron_fraction_low};
  variable_fixer(&specific_internal_energy, &electron_fraction_low, &pressure,
                 &specific_enthalpy, density, equation_of_state);

  CHECK_ITERABLE_APPROX(electron_fraction_low, expected_electron_fraction_low);
}

void test_variable_fixer() {
  // Test for representative 1-d equation of state

  const bool enable_param_delep = true;

  const double hi_dens_cutoff = 2.0e-10;
  const double lo_dens_cutoff = 1.0e-10;

  const double ye_at_hi_dens = 0.285;
  const double ye_at_lo_dens = 0.5;

  const double ye_magnitude_scale = 0.035;

  // high density scale, low density scale, Ye(hi dens), Ye(low dens), Ye
  // correction scale
  const VariableFixing::ParameterizedDeleptonization variable_fixer{
      enable_param_delep, hi_dens_cutoff, lo_dens_cutoff,
      ye_at_hi_dens,      ye_at_lo_dens,  ye_magnitude_scale};

  const EquationsOfState::PolytropicFluid<true> polytrope{1.0, 2.0};

  // catch error messages
  CHECK_THROWS_WITH(([&hi_dens_cutoff, &lo_dens_cutoff, &ye_at_hi_dens,
                      &ye_at_lo_dens, &ye_magnitude_scale]() {
                      const VariableFixing::ParameterizedDeleptonization
                          variable_fixer_dens_error{
                              enable_param_delep,    hi_dens_cutoff,
                              10.0 * lo_dens_cutoff, ye_at_hi_dens,
                              ye_at_lo_dens,         ye_magnitude_scale};
                    })(),
                    Catch::Contains("The high density scale"));

  CHECK_THROWS_WITH(([&hi_dens_cutoff, &lo_dens_cutoff, &ye_at_lo_dens,
                      &ye_magnitude_scale]() {
                      const VariableFixing::ParameterizedDeleptonization
                          variable_fixer_dens_error{
                              enable_param_delep, hi_dens_cutoff,
                              lo_dens_cutoff,     10.0 * ye_at_lo_dens,
                              ye_at_lo_dens,      ye_magnitude_scale};
                    })(),
                    Catch::Contains("The Ye at high density("));

  // check expected values
  test_variable_fixer(variable_fixer, polytrope);
  test_serialization(variable_fixer);

  {
    INFO("Test 1D/barotropic EOS");
    const auto fixer_from_options = TestHelpers::test_creation<
        VariableFixing::ParameterizedDeleptonization>(
        "Enable: true\n"
        "HighDensityScale: 2.0e-10\n"
        "LowDensityScale: 1.0e-10\n"
        "ElectronFractionAtHighDensity: 0.285\n"
        "ElectronFractionAtLowDensity: 0.5\n"
        "ElectronFractionCorrectionScale: 0.035");

    test_serialization(fixer_from_options);
    test_variable_fixer(fixer_from_options, polytrope);
  }

  {
    INFO("Test 2D EOS");
    const auto fixer_from_options = TestHelpers::test_creation<
        VariableFixing::ParameterizedDeleptonization>(
        "Enable: true\n"
        "HighDensityScale: 4.0e-9\n"
        "LowDensityScale: 3.0e-9\n"
        "ElectronFractionAtHighDensity: 0.2\n"
        "ElectronFractionAtLowDensity: 0.49\n"
        "ElectronFractionCorrectionScale: 0.05");

    EquationsOfState::IdealFluid<true> ideal_fluid{5.0 / 3.0};
    test_serialization(fixer_from_options);
    test_variable_fixer(fixer_from_options, ideal_fluid);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.ParameterizedDeleptonization",
                  "[VariableFixing][Unit]") {
  test_variable_fixer();
}
