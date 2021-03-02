// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
void test() {
  CAPTURE(Dim);
  helpers::test_periodic_condition<
      domain::BoundaryConditions::Periodic<
          GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>>,
      GeneralizedHarmonic::BoundaryConditions::BoundaryCondition<Dim>>(
      "Periodic:\n");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GeneralizedHarmonic.BoundaryConditions.Periodic",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
