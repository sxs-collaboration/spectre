// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

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
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
  const auto specific_internal_energy =
      eos.specific_internal_energy_from_density(rest_mass_density);
  const auto specific_enthalpy =
      eos.specific_enthalpy_from_density(rest_mass_density);

  const auto lapse = gr_helper::random_lapse(nn_gen, used_for_size);
  const auto shift = gr_helper::random_shift<3>(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);

  const auto magnetic_field = helper::random_magnetic_field(
      nn_gen, eos.pressure_from_density(rest_mass_density), spatial_metric);
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
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const auto& eos_base =
        static_cast<const EquationsOfState::EquationOfState<true, 1>&>(eos);
    Approx custom_approx = Approx::custom().epsilon(4.0e-12);
    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds(
            rest_mass_density, specific_internal_energy, specific_enthalpy,
            spatial_velocity, lorentz_factor, magnetic_field, lapse, shift,
            spatial_metric, normal, eos_base),
        (pypp::call<std::array<DataVector, 9>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            alfven_speed_squared, normal)), custom_approx);
  }
}

struct SomeEosType {
  static constexpr size_t thermodynamic_dim = 3;
};

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

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute<SomeEosType>>(
      "CharacteristicSpeeds");
}
