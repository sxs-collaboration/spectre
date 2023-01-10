// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Punctures/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Punctures/Sources.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

void test_equations(const DataVector& used_for_size) {
  const double eps = 1.e-12;
  const auto seed = std::random_device{}();
  const double fill_result_tensors = 0.;
  pypp::check_with_random_values<3>(
      &Punctures::add_sources, "Equations", {"sources"},
      {{{-1., 1.}, {-1., 1.}, {-1., 1.}}}, used_for_size, eps, seed,
      fill_result_tensors);
  pypp::check_with_random_values<4>(
      &Punctures::add_linearized_sources, "Equations", {"linearized_sources"},
      {{{-1., 1.}, {-1., 1.}, {-1., 1.}, {-1., 1.}}}, used_for_size, eps, seed,
      fill_result_tensors);
}

void test_computers(const DataVector& used_for_size) {
  using system = Punctures::FirstOrderSystem;
  static_assert(
      tt::assert_conforms_to_v<system, elliptic::protocols::FirstOrderSystem>);
  TestHelpers::elliptic::test_first_order_fluxes_computer<system>(
      used_for_size);
  TestHelpers::elliptic::test_first_order_sources_computer<system, false>(
      used_for_size);
  TestHelpers::elliptic::test_first_order_sources_computer<system, true>(
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Punctures", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/Systems/Punctures"};
  GENERATE_UNINITIALIZED_DATAVECTOR;
  test_equations(dv);
  test_computers(dv);
}
