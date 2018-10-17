// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Dim, typename DataType>
void test_gh_fluxes(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &GeneralizedHarmonic::ComputeNormalDotFluxes<Dim>::apply, "TestFunctions",
      {"spacetime_metric_normal_dot_flux", "pi_normal_dot_flux",
       "phi_dot_flux"},
      {{{-1.0, 1.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.NormalDotFluxes",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GeneralizedHarmonic/"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_gh_fluxes, (1, 2, 3));
}
