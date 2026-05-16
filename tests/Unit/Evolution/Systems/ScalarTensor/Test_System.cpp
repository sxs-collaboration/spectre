// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/ScalarTensor/System.hpp"

SPECTRE_TEST_CASE("Unit.ScalarTensor.System.Name", "[Unit][Evolution]") {
  CHECK(ScalarTensor::System::name() == "ScalarTensor");
}
