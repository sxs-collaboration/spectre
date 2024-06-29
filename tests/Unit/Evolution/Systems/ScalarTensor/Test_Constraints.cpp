// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/ScalarTensor/Constraints.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarTensor.Constraints",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_compute_tag<
      ScalarTensor::Tags::FConstraintCompute<3, Frame::Inertial>>(
      "FConstraint");
}
