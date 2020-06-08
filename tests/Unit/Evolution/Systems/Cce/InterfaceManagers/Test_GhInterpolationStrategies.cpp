// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Evolution/Systems/Cce/InterfaceManagers/GhInterpolationStrategies.hpp"
#include "Utilities/Literals.hpp"

namespace Cce::InterfaceManagers {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.GhInterpolationStrategies",
                  "[Unit][Cce]") {
  auto test_box = db::create<db::AddSimpleTags<::Tags::TimeStepId>>(
      TimeStepId{true, 0_st, {{0.0, 0.1}, {1, 2}}});
  CHECK(should_interpolate_for_strategy(test_box,
                                        InterpolationStrategy::EveryStep));
  CHECK(should_interpolate_for_strategy(test_box,
                                        InterpolationStrategy::EverySubstep));
  test_box = db::create<db::AddSimpleTags<::Tags::TimeStepId>>(
      TimeStepId{true, 0_st, {{0.0, 0.1}, {1, 2}}, 1_st, {{0.0, 0.1}, {2, 2}}});
  CHECK_FALSE(should_interpolate_for_strategy(test_box,
                                        InterpolationStrategy::EveryStep));
  CHECK(should_interpolate_for_strategy(test_box,
                                        InterpolationStrategy::EverySubstep));
}
}  // namespace Cce::InterfaceManagers
