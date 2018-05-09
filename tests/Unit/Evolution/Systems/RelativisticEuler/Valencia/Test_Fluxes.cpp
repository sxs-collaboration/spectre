// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Fluxes.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_fluxes(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &RelativisticEuler::Valencia::fluxes<Dim>, "TestFunctions",
      {"tilde_d_flux", "tilde_tau_flux", "tilde_s_flux"}, {{{0.0, 1.0}}},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Fluxes",
                  "[Unit][RelativisticEuler]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  const DataVector dv(5);
  test_fluxes<1>(dv);
  test_fluxes<2>(dv);
  test_fluxes<3>(dv);
}
