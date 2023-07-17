// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/TimeStepper.hpp"

namespace {
struct DummyType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Tags.TimeStepper", "[Unit][Time]") {
  TestHelpers::db::test_base_tag<Tags::TimeStepper<>>("TimeStepper");
  TestHelpers::db::test_simple_tag<Tags::TimeStepper<DummyType>>("TimeStepper");
}
