// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ForceFree/ElectricCurrentDensity.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.ElectricCurrentDensity",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(&ForceFree::ComputeDriftTildeJ::apply,
                                    "ElectricCurrentDensity", {"tilde_j_drift"},
                                    {{{-1.0, 1.0}}}, DataVector{5}, 1.0e-10);

  pypp::check_with_random_values<1>(
      &ForceFree::ComputeParallelTildeJ::apply, "ElectricCurrentDensity",
      {"tilde_j_parallel"}, {{{-1.0, 1.0}}}, DataVector{5}, 1.0e-10);

  pypp::check_with_random_values<1>(&ForceFree::Tags::ComputeTildeJ::function,
                                    "ElectricCurrentDensity", {"tilde_j"},
                                    {{{-1.0, 1.0}}}, DataVector{5}, 1.0e-10);
}
