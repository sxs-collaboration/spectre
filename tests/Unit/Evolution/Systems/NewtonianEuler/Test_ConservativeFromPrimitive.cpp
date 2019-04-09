// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_conservative_from_primitive(const DataVector& used_for_size) {
  pypp::check_with_random_values<3>(
      &NewtonianEuler::ConservativeFromPrimitive<Dim>::apply, "TestFunctions",
      {"mass_density_cons", "momentum_density", "energy_density"},
      {{{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.ConservativeFromPrimitive",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_conservative_from_primitive, (1, 2, 3))
}
