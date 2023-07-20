// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarCharge.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.ScalarCharge",
                  "[Unit][PointwiseFunctions]") {
  TestHelpers::db::test_simple_tag<
      ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrand>(
      "ScalarChargeIntegrand");
  TestHelpers::db::test_compute_tag<
      ScalarTensor::StrahlkorperScalar::Tags::ScalarChargeIntegrandCompute>(
      "ScalarChargeIntegrand");
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor"};

  pypp::check_with_random_values<1>(&ScalarTensor::scalar_charge_integrand,
                                    "ScalarCharge", {"scalar_charge_integrand"},
                                    {{{1.0e-2, 0.5}}}, DataVector{5});
}
