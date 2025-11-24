// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ForceFree/ElectromagneticVariables.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.ElectromagneticVariables",
                  "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(
      ForceFree::Tags::ElectricFieldCompute::function,
      "ElectromagneticVariables", {"electric_field_compute"}, {{{-1.0, 1.0}}},
      DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::MagneticFieldCompute::function,
      "ElectromagneticVariables", {"magnetic_field_compute"}, {{{-1.0, 1.0}}},
      DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::ChargeDensityCompute::function,
      "ElectromagneticVariables", {"charge_density_compute"}, {{{-1.0, 1.0}}},
      DataVector{5});

  pypp::check_with_random_values<1>(
      ForceFree::Tags::ElectricCurrentDensityCompute::function,
      "ElectromagneticVariables", {"electric_current_density_compute"},
      {{{-1.0, 1.0}}}, DataVector{5});
}
