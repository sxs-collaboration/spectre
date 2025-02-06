// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/ExitCode.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.ExitCode", "[Parallel][Unit]") {
  CHECK(get_output(Parallel::ExitCode::Complete) == "0 (Complete)");
  CHECK(get_output(Parallel::ExitCode::Abort) == "1 (Abort)");
  CHECK(get_output(Parallel::ExitCode::ContinueFromCheckpoint) ==
        "2 (ContinueFromCheckpoint)");
  CHECK_THROWS_WITH(get_output(static_cast<Parallel::ExitCode>(3)),
                    Catch::Matchers::ContainsSubstring("Unknown exit code: 3"));
}
