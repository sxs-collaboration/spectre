// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/SimpleProfilerData.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Parallel.Tags.SimpleProfilerData", "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<Parallel::Tags::SimpleProfilerData>(
      "SimpleProfilerData");
}
}  // namespace
