// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/EquationsOfState/SpecificEnthalpy.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <typename DataType>
void test_specific_enthalpy(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      &EquationsOfState::specific_enthalpy<DataType>, "TestFunctions",
      "specific_enthalpy", {{{0.01, 1.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.EquationsOfState.SpecificEnthalpy",
                  "[Unit][EquationsOfState]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/EquationsOfState"};

  test_specific_enthalpy(std::numeric_limits<double>::signaling_NaN());
  test_specific_enthalpy(DataVector(5));
}
