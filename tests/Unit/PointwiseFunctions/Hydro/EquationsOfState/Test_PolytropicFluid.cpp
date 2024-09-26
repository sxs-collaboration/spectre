// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <pup.h>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/EquationsOfState/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic2D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Barotropic3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
// check that pressure equals energy density at upper bound of specific
// internal energy for adiabatic index > 2
void check_dominant_energy_condition_at_bound() {
  const auto seed = std::random_device{}();
  MAKE_GENERATOR(generator, seed);
  CAPTURE(seed);
  auto distribution = std::uniform_real_distribution<>{2.0, 3.0};
  const double polytropic_exponent = distribution(generator);
  const auto eos =
      EquationsOfState::PolytropicFluid<true>{100.0, polytropic_exponent};
  const Scalar<double> rest_mass_density{eos.rest_mass_density_upper_bound()};
  const double specific_internal_energy =
      get(eos.specific_internal_energy_from_density(rest_mass_density));
  const double pressure =
      get(eos.pressure_from_density(Scalar<double>{rest_mass_density}));
  const double energy_density =
      get(rest_mass_density) * (1.0 + specific_internal_energy);
  CAPTURE(rest_mass_density);
  CAPTURE(specific_internal_energy);
  CHECK(approx(pressure) == energy_density);
}

template <bool IsRelativistic>
void check_bounds() {
  const auto eos =
      EquationsOfState::PolytropicFluid<IsRelativistic>{100.0, 1.5};
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(0.0 == eos.specific_internal_energy_lower_bound());
  if constexpr (IsRelativistic) {
    CHECK(1.0 == eos.specific_enthalpy_lower_bound());
  } else {
    CHECK(0.0 == eos.specific_enthalpy_lower_bound());
  }
  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound());
  CHECK(eos.baryon_mass() ==
        approx(hydro::units::geometric::default_baryon_mass));
}

}  // namespace
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.PolytropicFluid",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  register_derived_classes_with_charm<EoS::EquationOfState<true, 1>>();
  register_derived_classes_with_charm<EoS::EquationOfState<false, 1>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  TestHelpers::EquationsOfState::test_get_clone(
      EoS::PolytropicFluid<true>(100.0, 2.0));

  const auto eos = EoS::PolytropicFluid<true>{100.0, 2.0};
  const auto other_eos = EoS::PolytropicFluid<true>{4.0, 2.0};
  const auto other_type_eos = EoS::DarkEnergyFluid<true>{0.5};
  CHECK(eos == eos);
  CHECK(eos != other_eos);
  CHECK(eos != other_type_eos);
  CHECK(*eos.promote_to_3d_eos() ==
        EoS::Barotropic3D<EoS::PolytropicFluid<true>>(eos));
  CHECK(*eos.promote_to_2d_eos() ==
        EoS::Barotropic2D<EoS::PolytropicFluid<true>>(eos));
  CHECK(eos.is_barotropic());
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<true>{100.0, 2.0},
                                       "PolytropicFluid", "polytropic",
                                       d_for_size, 100.0, 2.0);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<true>{134.0, 1.5},
                                       "PolytropicFluid", "polytropic",
                                       dv_for_size, 134.0, 1.5);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<false>{121.0, 1.2},
                                       "PolytropicFluid", "polytropic",
                                       d_for_size, 121.0, 1.2);
  TestHelpers::EquationsOfState::check(EoS::PolytropicFluid<false>{117.0, 1.12},
                                       "PolytropicFluid", "polytropic",
                                       dv_for_size, 117.0, 1.12);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 100.0\n"
           "  PolytropicExponent: 2.0\n"}),
      "PolytropicFluid", "polytropic", d_for_size, 100.0, 2.0);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 134.0\n"
           "  PolytropicExponent: 1.5\n"}),
      "PolytropicFluid", "polytropic", dv_for_size, 134.0, 1.5);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 121.0\n"
           "  PolytropicExponent: 1.2\n"}),
      "PolytropicFluid", "polytropic", d_for_size, 121.0, 1.2);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<false, 1>>>(
          {"PolytropicFluid:\n"
           "  PolytropicConstant: 117.0\n"
           "  PolytropicExponent: 1.12\n"}),
      "PolytropicFluid", "polytropic", dv_for_size, 117.0, 1.12);

  check_bounds<true>();
  check_bounds<false>();
  check_dominant_energy_condition_at_bound();
}
