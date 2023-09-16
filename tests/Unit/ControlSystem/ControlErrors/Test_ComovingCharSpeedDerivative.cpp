// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ControlSystem/ControlErrors/Size/ComovingCharSpeedDerivative.hpp"
#include "DataStructures/DataVector.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {

template <typename DataType>
void test_comoving_char_speed_derivative(const DataType& used_for_size) {
  pypp::check_with_random_values<14>(
      &control_system::size::comoving_char_speed_derivative,
      "ComovingCharSpeedDerivative", {{"comoving_char_speed_derivative"}},
      {{
          {-0.5, 0.},
          {-1., 1.},
          {2.2, 2.5},
          {-1., 1.},
          {1.9, 2.1},
          {-0.8, 0.8},
          {-1., 1.},
          {1., 2.},
          {-1., 1.},
          {1., 2.},
          {-1., 1.},
          {-1., 1.},
          {-1., 1.},
          {-1., 1.},
      }},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.ControlSystem.ControlErrors.ComovingCharSpeedDerivative",
    "[Domain][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "ControlSystem/ControlErrors/");
  DataVector used_for_size(3);
  test_comoving_char_speed_derivative(used_for_size);
}
