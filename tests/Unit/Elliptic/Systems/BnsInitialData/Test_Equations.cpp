// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/BnsInitialData/Equations.hpp"
#include "Elliptic/Systems/BnsInitialData/FirstOrderSystem.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace helpers = TestHelpers::elliptic;

namespace {
void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(&BnsInitialData::potential_fluxes,
                                    "Equations", {"potential_fluxes"},
                                    {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(&BnsInitialData::fluxes_on_face,
                                    "Equations", {"fluxes_on_face"},
                                    {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &BnsInitialData::add_potential_sources, "Equations",
      {"add_potential_sources"}, {{{0., 1.}}}, used_for_size, 1.e-12, {}, 0.);
}

void test_computers(const DataVector& used_for_size) {
  using system = BnsInitialData::FirstOrderSystem;
  static_assert(
      tt::assert_conforms_to_v<system, elliptic::protocols::FirstOrderSystem>);
  helpers::test_first_order_fluxes_computer<system>(used_for_size);
  helpers::test_first_order_sources_computer<system>(used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.BnsInitialData", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/Systems/BnsInitialData"};

  DataVector used_for_size{5};
  test_equations(used_for_size);
  test_computers(used_for_size);
}
