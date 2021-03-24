// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Time/SelfStart.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"

SPECTRE_TEST_CASE("Unit.Time.SelfStart", "[Unit][Time][Actions]") {
  CHECK(SelfStart::is_self_starting(
      TimeStepId{true, -1, Time{{0.0, 1.0}, {1, 5}}}));
  CHECK(SelfStart::is_self_starting(
      TimeStepId{true, -5, Time{{0.5, 0.6}, {4, 9}}}));
  CHECK_FALSE(SelfStart::is_self_starting(
      TimeStepId{true, 0, Time{{0.0, 1.0}, {1, 5}}}));
  CHECK_FALSE(SelfStart::is_self_starting(
      TimeStepId{true, 5, Time{{0.5, 0.6}, {4, 9}}}));
}
