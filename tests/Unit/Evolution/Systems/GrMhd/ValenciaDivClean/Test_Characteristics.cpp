// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace {

void test_characteristic_speeds(const DataVector& /*used_for_size*/) noexcept {
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  // const double max_value = 1.0 / sqrt(3);
  // pypp::check_with_random_values<7>(
  //     &grmhd::ValenciaDivClean::characteristic_speeds<3>, "TestFunctions",
  //     "characteristic_speeds",
  //     {{{0.0, 1.0},
  //       {-1.0, 1.0},
  //       {-max_value, max_value},
  //       {0.0, 1.0},
  //       {0.0, 1.0},
  //       {0.0, 1.0},
  //       {-max_value, max_value}}},
  //     used_for_size);
}

void test_with_normal_along_coordinate_axes(
    const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);
  std::uniform_real_distribution<> small_distribution(0.0, 0.1);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
  const auto specific_internal_energy =
      eos.specific_internal_energy_from_density(rest_mass_density);
  const auto specific_enthalpy =
      eos.specific_enthalpy_from_density(rest_mass_density);

  const auto lapse = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  const auto spatial_velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, make_not_null(&small_distribution), used_for_size);
  const auto spatial_metric = make_with_random_values<tnsr::ii<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);
  const auto lorentz_factor = hydro::lorentz_factor(spatial_velocity_squared);

  const auto magnetic_field = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  const auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(spatial_velocity, magnetic_field, spatial_metric);
  const DataVector comoving_magnetic_field_squared =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));
  Scalar<DataVector> alfven_speed_squared{
      comoving_magnetic_field_squared /
      (comoving_magnetic_field_squared +
       get(rest_mass_density) * get(specific_enthalpy))};
  Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(specific_enthalpy)};

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto normal = euclidean_basis_vector(direction, used_for_size);

    const auto& eos_base =
        static_cast<const EquationsOfState::EquationOfState<true, 1>&>(eos);
    const auto computed = grmhd::ValenciaDivClean::characteristic_speeds(
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        spatial_velocity, lorentz_factor, magnetic_field, lapse, shift,
        spatial_metric, normal, eos_base);

    const auto expected = pypp::call<std::array<DataVector, 9>>(
        "TestFunctions", "characteristic_speeds", lapse, shift,
        spatial_velocity, spatial_velocity_squared, sound_speed_squared,
        alfven_speed_squared, normal);
    CHECK_ITERABLE_APPROX(computed, expected);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  const DataVector dv(5);
  test_characteristic_speeds(dv);
  // Test with aligned normals to check the code works
  // with vector components being 0.
  test_with_normal_along_coordinate_axes(dv);
}
