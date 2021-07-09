// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Constraints.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Constraints",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_compute_tag<
      grmhd::GhValenciaDivClean::Tags::FConstraintCompute<3, Frame::Inertial>>(
      "FConstraint");
}
