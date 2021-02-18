// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.LongitudinalOperator",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions/Xcts"};
  const DataVector used_for_size{5};
  pypp::check_with_random_values<1>(
      &Xcts::longitudinal_operator<DataVector>, "LongitudinalOperator",
      {"longitudinal_operator"}, {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &Xcts::longitudinal_operator_flat_cartesian<DataVector>,
      "LongitudinalOperator", {"longitudinal_operator_flat_cartesian"},
      {{{-1., 1.}}}, used_for_size);
}
