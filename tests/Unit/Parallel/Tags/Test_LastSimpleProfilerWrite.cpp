// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/LastSimpleProfilerWrite.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Parallel.Tags.LastSimpleProfilerWrite",
                  "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<Parallel::Tags::LastSimpleProfilerWrite>(
      "LastSimpleProfilerWrite");
}
}  // namespace
