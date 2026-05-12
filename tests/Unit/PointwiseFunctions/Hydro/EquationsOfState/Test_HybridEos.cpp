// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/HybridEos.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
template <bool IsRelativistic>
void check_random_polytrope() {
  register_derived_classes_with_charm<
      EquationsOfState::EquationOfState<true, 2>>();
  register_derived_classes_with_charm<
      EquationsOfState::EquationOfState<false, 2>>();
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);
  TestHelpers::EquationsOfState::check(
      EquationsOfState::HybridEos<
          EquationsOfState::PolytropicFluid<IsRelativistic>>{
          EquationsOfState::PolytropicFluid<IsRelativistic>{100.0, 4.0 / 3.0},
          5.0 / 3.0, 1.e-3},
      "HybridEos", "hybrid_polytrope", d_for_size, 100.0, 4.0 / 3.0, 5.0 / 3.0,
      1.e-3);
  TestHelpers::EquationsOfState::check(
      EquationsOfState::HybridEos<
          EquationsOfState::PolytropicFluid<IsRelativistic>>{
          EquationsOfState::PolytropicFluid<IsRelativistic>{100.0, 4.0 / 3.0},
          5.0 / 3.0, 1.e-3},
      "HybridEos", "hybrid_polytrope", dv_for_size, 100.0, 4.0 / 3.0, 5.0 / 3.0,
      1.e-3);
}

template <bool IsRelativistic>
void check_exact_polytrope() {
  EquationsOfState::PolytropicFluid<IsRelativistic> cold_eos{3.0, 2.0};
  const Scalar<double> rho{4.0};
  const auto p_c = cold_eos.pressure_from_density(rho);
  CHECK(get(p_c) == 48.0);
  const auto eps_c = cold_eos.specific_internal_energy_from_density(rho);
  CHECK(get(eps_c) == 12.0);
  const auto h_c = IsRelativistic
                       ? hydro::relativistic_specific_enthalpy(rho, eps_c, p_c)
                       : Scalar<double>{get(eps_c) + get(p_c) / get(rho)};
  CHECK(get(h_c) == (IsRelativistic ? 25.0 : 24.0));
  const auto chi_c = cold_eos.chi_from_density(rho);
  CHECK(get(chi_c) == 24.0);
  const auto p_c_kappa_c_over_rho_sq =
      cold_eos.kappa_times_p_over_rho_squared_from_density(rho);
  CHECK(get(p_c_kappa_c_over_rho_sq) == 0.0);
  const auto c_s_sq = (get(chi_c) + get(p_c_kappa_c_over_rho_sq)) / get(h_c);
  CHECK(c_s_sq == (IsRelativistic ? 0.96 : 1.0));
  EquationsOfState::HybridEos<EquationsOfState::PolytropicFluid<IsRelativistic>>
      eos{cold_eos, 1.5};
  TestHelpers::EquationsOfState::test_get_clone(eos);

  const EquationsOfState::HybridEos<EquationsOfState::PolytropicFluid<true>>
      other_eos{{100.0, 2.0}, 1.4};
  const auto other_type_eos =
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0};
  CHECK(eos == eos);
  CHECK(eos != other_eos);
  CHECK(eos != other_type_eos);
  CHECK(*eos.promote_to_3d_eos() ==
        EquationsOfState::Equilibrium3D<EquationsOfState::HybridEos<
            EquationsOfState::PolytropicFluid<IsRelativistic>>>(eos));
  const Scalar<double> eps{18.0};
  const auto p = eos.pressure_from_density_and_energy(rho, eps);
  CHECK(get(p) == 60.0);
  const auto h = IsRelativistic
                     ? hydro::relativistic_specific_enthalpy(rho, eps, p)
                     : Scalar<double>{get(eps) + get(p) / get(rho)};
  CHECK(get(h) == (IsRelativistic ? 34.0 : 33.0));
  CHECK(get(eos.pressure_from_density_and_enthalpy(rho, h)) == 60.0);
  CHECK(get(eos.specific_internal_energy_from_density_and_pressure(rho, p)) ==
        get(eps));
  const auto chi = eos.chi_from_density_and_energy(rho, eps);
  CHECK(get(chi) == 21.0);
  const auto p_kappa_over_rho_sq =
      eos.kappa_times_p_over_rho_squared_from_density_and_energy(rho, eps);
  CHECK(get(p_kappa_over_rho_sq) == 7.5);
  CHECK(not eos.is_barotropic());
}

template <bool IsRelativistic, bool NonZeroMinTemp>
void check_bounds() {
  const auto cold_eos =
      EquationsOfState::PolytropicFluid<IsRelativistic>{100.0, 1.5};
  auto min_temperature = 0.0;
  if constexpr (NonZeroMinTemp) {
    const auto seed = std::random_device{}();
    MAKE_GENERATOR(generator, seed);
    CAPTURE(seed);
    auto distribution = std::uniform_real_distribution<>{1.e-15, 1.e-5};
    min_temperature = distribution(generator);
  }
  const auto thermal_adiabatic_index = 1.5;
  CAPTURE(min_temperature);
  CAPTURE(thermal_adiabatic_index);
  const EquationsOfState::HybridEos<
      EquationsOfState::PolytropicFluid<IsRelativistic>>
      eos{cold_eos, thermal_adiabatic_index, min_temperature};
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(min_temperature == eos.temperature_lower_bound());
  CHECK(
      get(cold_eos.specific_internal_energy_from_density(Scalar<double>{1.0})) +
          min_temperature / (thermal_adiabatic_index - 1) ==
      eos.specific_internal_energy_lower_bound(1.0));
  if constexpr (IsRelativistic) {
    CHECK(1.0 + (thermal_adiabatic_index * min_temperature) /
                    (thermal_adiabatic_index - 1.0) ==
          eos.specific_enthalpy_lower_bound());
  } else {
    CHECK((thermal_adiabatic_index * min_temperature) /
              (thermal_adiabatic_index - 1.0) ==
          eos.specific_enthalpy_lower_bound());
  }
  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound(1.0));
  CHECK(eos.baryon_mass() ==
        approx(hydro::units::geometric::default_baryon_mass));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.HybridEos",
                  "[Unit][EquationsOfState]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  check_random_polytrope<true>();
  check_random_polytrope<false>();
  check_exact_polytrope<true>();
  check_exact_polytrope<false>();
  check_bounds<true, true>();
  check_bounds<true, false>();
  check_bounds<false, true>();
  check_bounds<false, false>();
}
