// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
SPECTRE_TEST_CASE("Unit.Parallel.AlgorithmExecution", "[Unit][Parallel]") {
#define CHECK_NAME(name) \
  CHECK(get_output(Parallel::AlgorithmExecution::name) == #name)

  CHECK_NAME(Continue);
  CHECK_NAME(Retry);
  CHECK_NAME(Pause);
  CHECK_NAME(Halt);
#undef CHECK_NAME
}
}  // namespace
