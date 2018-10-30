// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/AlfvenSpeedSquared.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace hydro {
namespace {
template <typename DataType>
void test_alfven_speed_squared(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(&alfven_speed_squared<DataType>,
                                    "TestFunctions", "alfven_speed_squared",
                                    {{{0.0, 10.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.AlfvenSpeedSquared",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");

  test_alfven_speed_squared(std::numeric_limits<double>::signaling_NaN());
  test_alfven_speed_squared(DataVector(5));
}
}  // namespace hydro
