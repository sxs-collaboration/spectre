// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.Burgers.Fluxes", "[Unit][Burgers]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/Burgers"};

  pypp::check_with_random_values<1>(&Burgers::Fluxes::apply, "Flux", {"flux"},
                                    {{{-1.0, 1.0}}}, DataVector{5});
}
