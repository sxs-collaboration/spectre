// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/Factory.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"

namespace helpers = TestHelpers::evolution::dg;

SPECTRE_TEST_CASE("Unit.Burgers.BoundaryConditions.Periodic",
                  "[Unit][Burgers]") {
  helpers::test_periodic_condition<
      domain::BoundaryConditions::Periodic<
          Burgers::BoundaryConditions::BoundaryCondition>,
      Burgers::BoundaryConditions::BoundaryCondition>("Periodic:\n");
}
