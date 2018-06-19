// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim, typename DataType>
void test_conservative_from_primitive(const DataType& used_for_size) {
  pypp::check_with_random_values<3>(
      &NewtonianEuler::conservative_from_primitive<Dim, DataType>,
      "TestFunctions", {"momentum_density", "energy_density"},
      {{{-1.0, 1.0}, {-2.0, 2.0}, {-3.0, 3.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.ConservativeFromPrimitive",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_conservative_from_primitive, (1, 2, 3))
}
