// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryConditions.Periodic",
                  "[Unit][Evolution]") {
  helpers::test_periodic_condition<
      domain::BoundaryConditions::Periodic<
          grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition>,
      grmhd::ValenciaDivClean::BoundaryConditions::BoundaryCondition>(
      "Periodic:\n");
}
