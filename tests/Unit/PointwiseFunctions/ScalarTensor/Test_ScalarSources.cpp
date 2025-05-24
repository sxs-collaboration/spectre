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
#include "PointwiseFunctions/ScalarTensor/ScalarSource.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.ScalarSource",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/ScalarTensor"};

  pypp::check_with_random_values<1>(&ScalarTensor::mass_source, "Sources",
                                    {"mass_source"}, {{{1.0e-2, 0.5}}},
                                    DataVector{5});

  pypp::check_with_random_values<1>(
      &ScalarTensor::add_scalar_source_to_dt_pi_scalar, "Sources",
      {"add_scalar_source_to_dt_pi_scalar"}, {{{1.0e-2, 0.5}}}, DataVector{5},
      1.0e-12, std::random_device{}(), 0.1234);
}
