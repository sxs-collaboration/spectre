// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

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

  const double d = std::numeric_limits<double>::signaling_NaN();
  test_conservative_from_primitive<1>(d);
  test_conservative_from_primitive<2>(d);
  test_conservative_from_primitive<3>(d);

  const DataVector dv(5);
  test_conservative_from_primitive<1>(dv);
  test_conservative_from_primitive<2>(dv);
  test_conservative_from_primitive<3>(dv);
}
