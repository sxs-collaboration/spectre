// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"

namespace {
template <size_t Dim>
void test() {
  CAPTURE(Dim);
  TestHelpers::evolution::dg::test_periodic_condition<
      domain::BoundaryConditions::Periodic<
          SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<Dim>>,
      SecondOrderScalarWave::BoundaryConditions::BoundaryCondition<Dim>>(
      "Periodic:\n");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SecondOrderScalarWave.BoundaryConditions.Periodic",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
