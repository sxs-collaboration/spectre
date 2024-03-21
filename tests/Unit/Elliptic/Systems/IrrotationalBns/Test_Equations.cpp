// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/IrrotationalBns/Equations.hpp"
#include "Elliptic/Systems/IrrotationalBns/FirstOrderSystem.hpp"
#include "Elliptic/Systems/IrrotationalBns/Geometry.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace helpers = TestHelpers::elliptic;

namespace {
void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(&IrrotationalBns::flat_potential_fluxes,
                                    "Equations", {"flat_potential_fluxes"},
                                    {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(&IrrotationalBns::curved_potential_fluxes,
                                    "Equations", {"curved_potential_fluxes"},
                                    {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &IrrotationalBns::add_curved_potential_sources, "Equations",
      {"add_curved_potential_sources"}, {{{0., 1.}}}, used_for_size, 1.e-12, {},
      0.);
  pypp::check_with_random_values<1>(&IrrotationalBns::auxiliary_fluxes,
                                    "Equations", {"auxiliary_fluxes"},
                                    {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &IrrotationalBns::add_auxiliary_sources_without_flux_christoffels,
      "Equations", {"add_auxiliary_sources_without_flux_christoffels"},
      {{{0., 1.}}}, used_for_size, 1.e-12, {}, 0.);
  pypp::check_with_random_values<1>(
      &IrrotationalBns::add_auxiliary_source_flux_christoffels, "Equations",
      {"add_auxiliary_source_flux_christoffels"}, {{{0., 1.}}}, used_for_size,
      1.e-12, {}, 0.);
}

template <IrrotationalBns::Geometry BackgroundGeometry>
void test_computers(const DataVector& used_for_size) {
  using system = IrrotationalBns::FirstOrderSystem<BackgroundGeometry>;
  static_assert(
      tt::assert_conforms_to_v<system, elliptic::protocols::FirstOrderSystem>);
  helpers::test_first_order_fluxes_computer<system>(used_for_size);
  helpers::test_first_order_sources_computer<system>(used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.IrrotationalBns", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/Systems/IrrotationalBns"};

  DataVector used_for_size{5};
  test_equations(used_for_size);
  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_computers,
                        (IrrotationalBns::Geometry::FlatCartesian,
                         IrrotationalBns::Geometry::Curved));
}
