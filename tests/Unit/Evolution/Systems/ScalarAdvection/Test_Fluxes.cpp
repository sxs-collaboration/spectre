// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_fluxes(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(&ScalarAdvection::Fluxes<Dim>::apply,
                                    "TestFunctions", {"compute_flux"},
                                    {{{-1.0, 1.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarAdvection.Fluxes",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarAdvection"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_fluxes, (1, 2))
}
