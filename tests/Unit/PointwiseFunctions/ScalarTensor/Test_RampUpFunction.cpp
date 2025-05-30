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
#include "PointwiseFunctions/ScalarTensor/RampUpFunction.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.RampUpFunction",
                  "[Unit][PointwiseFunctions]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor"};

  // Specify explicitly the template parameters to avoid ambiguous calls
  pypp::check_with_random_values<1, double (*)(double, double, double),
                                 DataVector, nullptr>(
      &ScalarTensor::nonic_ramp_function, "RampUpFunction",
      {"nonic_ramp_function"}, {{{-1.0, 1.0}}}, DataVector{5});
}
