// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/TimeAndPrevious.hpp"

SPECTRE_TEST_CASE("Unit.Time.Tags.TimeAndPrevious", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::TimeAndPrevious<0>>(
      "TimeAndPrevious0");
  TestHelpers::db::test_simple_tag<Tags::TimeAndPrevious<1>>(
      "TimeAndPrevious1");
  TestHelpers::db::test_compute_tag<Tags::TimeAndPreviousCompute<0>>(
      "TimeAndPrevious0");
  TestHelpers::db::test_compute_tag<Tags::TimeAndPreviousCompute<1>>(
      "TimeAndPrevious1");
}
