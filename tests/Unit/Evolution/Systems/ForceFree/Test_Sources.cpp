// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ForceFree/Sources.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Sources",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(&ForceFree::Sources::apply, "Sources",
                                    {"source_tilde_e", "source_tilde_b",
                                     "source_tilde_psi", "source_tilde_phi"},
                                    {{{0.0, 1.0}}}, DataVector{5});
}
