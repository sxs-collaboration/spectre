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
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/DarkEnergyFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

namespace {
void check_bounds() noexcept {
  const auto eos = EquationsOfState::DarkEnergyFluid<true>{1.0};
  CHECK(0.0 == eos.rest_mass_density_lower_bound());
  CHECK(-1.0 == eos.specific_internal_energy_lower_bound(1.0));
  CHECK(0.0 == eos.specific_enthalpy_lower_bound());
  const double max_double = std::numeric_limits<double>::max();
  CHECK(max_double == eos.rest_mass_density_upper_bound());
  CHECK(max_double == eos.specific_internal_energy_upper_bound(1.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.DarkEnergyFluid",
                  "[Unit][EquationsOfState]") {
  namespace EoS = EquationsOfState;
  Parallel::register_derived_classes_with_charm<
      EoS::EquationOfState<true, 2>>();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro/EquationsOfState/"};
  const double d_for_size = std::numeric_limits<double>::signaling_NaN();
  const DataVector dv_for_size(5);

  TestHelpers::EquationsOfState::check(EoS::DarkEnergyFluid<true>(1.0),
                                       "dark_energy_fluid", d_for_size, 1.0);
  TestHelpers::EquationsOfState::check(EoS::DarkEnergyFluid<true>(1.0 / 3.0),
                                       "dark_energy_fluid", dv_for_size,
                                       1.0 / 3.0);
  TestHelpers::EquationsOfState::check(EoS::DarkEnergyFluid<true>(1.0),
                                       "dark_energy_fluid", dv_for_size, 1.0);
  TestHelpers::EquationsOfState::check(EoS::DarkEnergyFluid<true>(1.0 / 3.0),
                                       "dark_energy_fluid", d_for_size,
                                       1.0 / 3.0);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 2>>>(
          {"DarkEnergyFluid:\n"
           "  ParameterW: 1.0\n"}),
      "dark_energy_fluid", d_for_size, 1.0);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 2>>>(
          {"DarkEnergyFluid:\n"
           "  ParameterW: 0.3333333333333333\n"}),
      "dark_energy_fluid", d_for_size, 1.0 / 3.0);

  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 2>>>(
          {"DarkEnergyFluid:\n"
           "  ParameterW: 1.0\n"}),
      "dark_energy_fluid", dv_for_size, 1.0);
  TestHelpers::EquationsOfState::check(
      TestHelpers::test_creation<
          std::unique_ptr<EoS::EquationOfState<true, 2>>>(
          {"DarkEnergyFluid:\n"
           "  ParameterW: 0.3333333333333333\n"}),
      "dark_energy_fluid", dv_for_size, 1.0 / 3.0);

  check_bounds();
}
