// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ForceFree/BoundaryConditions/Factory.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.ForceFree.BoundaryConditions.Periodic",
                  "[Unit][ForceFree]") {
  helpers::test_periodic_condition<
      domain::BoundaryConditions::Periodic<
          ForceFree::BoundaryConditions::BoundaryCondition>,
      ForceFree::BoundaryConditions::BoundaryCondition>("Periodic:\n");
}
