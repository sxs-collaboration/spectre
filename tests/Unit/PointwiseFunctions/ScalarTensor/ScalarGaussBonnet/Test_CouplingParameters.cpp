// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingParameters.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.Sgb.CouplingParamters",
                  "[Unit][PointwiseFunctions]") {
  const ScalarTensor::CouplingParameterOptions params{-2.0, 3.1, -40.2};
  CHECK(params == ScalarTensor::CouplingParameterOptions{-2.0, 3.1, -40.2});
  CHECK(params != ScalarTensor::CouplingParameterOptions{5.2, 3.1, -40.2});
  CHECK(params != ScalarTensor::CouplingParameterOptions{-2.0, 31.0, -40.2});
  CHECK(params != ScalarTensor::CouplingParameterOptions{-2.0, 3.1, 4.2});
  test_serialization(params);
  test_copy_semantics(params);
  const auto created_params =
      TestHelpers::test_creation<ScalarTensor::CouplingParameterOptions>(
          "Linear: -2.0\n"
          "Quadratic: 3.1\n"
          "Quartic: -40.2\n");
  CHECK(created_params == params);
}
