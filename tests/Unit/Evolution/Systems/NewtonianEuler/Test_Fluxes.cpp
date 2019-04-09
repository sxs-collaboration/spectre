// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_fluxes(const DataVector& used_for_size) {
  pypp::check_with_random_values<4>(
      &NewtonianEuler::ComputeFluxes<Dim>::apply, "TestFunctions",
      {"mass_density_cons_flux", "momentum_density_flux",
       "energy_density_flux"},
      {{{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Fluxes",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_fluxes, (1, 2, 3))
}
