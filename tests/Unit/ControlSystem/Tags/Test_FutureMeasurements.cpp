// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "ControlSystem/Tags/FutureMeasurements.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ControlSystem;
}  // namespace

SPECTRE_TEST_CASE("Unit.ControlSystem.Tags.FutureMeasurements",
                  "[Unit][ControlSystem]") {
  TestHelpers::db::test_simple_tag<
      control_system::Tags::FutureMeasurements<tmpl::list<ControlSystem>>>(
      "FutureMeasurements");
}
