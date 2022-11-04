// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ForceFree/SpatialCurrentDensity.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.SpatialCurrentDensity",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(
      &ForceFree::SpatialCurrentDensity::apply, "TestFunctions",
      {"spatial_current_density"}, {{{0.0, 1.0}}}, DataVector{5});
}
