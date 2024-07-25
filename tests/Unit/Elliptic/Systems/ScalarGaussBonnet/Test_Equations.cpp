// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/Equations.hpp"
#include "Elliptic/Systems/ScalarGaussBonnet/FirstOrderSystem.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace helpers = TestHelpers::elliptic;

namespace {

void test_equations(const DataVector& used_for_size) {
  const double eps = 1.e-12;
  const auto seed = std::random_device{}();
  const double fill_result_tensors = 0.;
  pypp::check_with_random_values<1>(
      &sgb::curved_fluxes, "Equations", {"curved_fluxes"}, {{{0., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &sgb::face_fluxes, "Equations", {"face_fluxes"}, {{{0., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &sgb::add_curved_sources, "Equations", {"add_curved_sources"},
      {{{0., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &sgb::add_GB_terms, "Equations", {"GB_source_term"}, {{{0., 1.}}},
      used_for_size, eps, seed, fill_result_tensors);
  pypp::check_with_random_values<1>(
      &sgb::add_linearized_GB_terms, "Equations", {"linearized_GB_source_term"},
      {{{0., 1.}}}, used_for_size, eps, seed, fill_result_tensors);
}

void test_computers(const DataVector& used_for_size) {
  using system = sgb::FirstOrderSystem;
  static_assert(
      tt::assert_conforms_to_v<system, elliptic::protocols::FirstOrderSystem>);
  helpers::test_first_order_fluxes_computer<system>(used_for_size);
  helpers::test_first_order_sources_computer<system>(used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.ScalarGaussBonnet",
                  "[Unit][Elliptic]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/Systems/ScalarGaussBonnet"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  test_equations(dv);
  test_computers(dv);
}
