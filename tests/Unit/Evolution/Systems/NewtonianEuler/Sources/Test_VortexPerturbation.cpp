// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_include <string>

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
// IWYU pragma: no_include "Utilities/Gsl.hpp"

namespace {

void test_sources(const DataVector& used_for_size) noexcept {
  pypp::check_with_random_values<6>(
      &NewtonianEuler::Sources::VortexPerturbation::apply, "VortexPerturbation",
      {"mass_density_source", "momentum_density_source",
       "energy_density_source"},
      {{{0., 1.E4},
        {-100., 100.},
        {0., 300.},
        {0., 1.E5},
        {-1., 1.},
        {-4., 4.}}},
      used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Sources.VortexPerturb",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler/Sources"};

  test_sources(DataVector(5));
}
