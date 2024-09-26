// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Elliptic/Protocols/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Elliptic/FirstOrderSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace helpers = TestHelpers::elliptic;

namespace {

template <typename DataType, size_t Dim>
void test_equations(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &Poisson::flat_cartesian_fluxes<DataType, Dim>, "Equations",
      {"flat_cartesian_fluxes"}, {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(&Poisson::curved_fluxes<DataType, Dim>,
                                    "Equations", {"curved_fluxes"},
                                    {{{0., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &Poisson::add_curved_sources<DataType, Dim>, "Equations",
      {"add_curved_sources"}, {{{0., 1.}}}, used_for_size, 1.e-12, {}, 0.);
}

template <typename DataType, size_t Dim, Poisson::Geometry BackgroundGeometry>
void test_computers(const DataVector& used_for_size) {
  CAPTURE(Dim);
  using system = Poisson::FirstOrderSystem<Dim, BackgroundGeometry, DataType>;
  static_assert(
      tt::assert_conforms_to_v<system, elliptic::protocols::FirstOrderSystem>);
  helpers::test_first_order_fluxes_computer<system>(used_for_size);
  helpers::test_first_order_sources_computer<system>(used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Poisson", "[Unit][Elliptic]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Elliptic/Systems/Poisson"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_equations, (DataVector, ComplexDataVector),
                        (1, 2, 3));
  CHECK_FOR_DATAVECTORS(
      test_computers, (DataVector, ComplexDataVector), (1, 2, 3),
      (Poisson::Geometry::FlatCartesian, Poisson::Geometry::Curved));
}
