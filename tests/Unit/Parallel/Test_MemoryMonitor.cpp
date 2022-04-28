// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/MemoryMonitor/Tags.hpp"

namespace {
template <typename Metavariables>
struct MockMemoryMonitor {};
struct TestMetavariables {};

void test_tags() {
  INFO("Test Tags");
  using holder_tag = mem_monitor::Tags::MemoryHolder;
  TestHelpers::db::test_simple_tag<holder_tag>("MemoryHolder");

  const std::string subpath =
      mem_monitor::subfile_name<MockMemoryMonitor<TestMetavariables>>();

  CHECK(subpath == "/MemoryMonitors/MockMemoryMonitor");
}

SPECTRE_TEST_CASE("Unit.Parallel.MemoryMonitor", "[Unit][Parallel]") {
  test_tags();
}
}  // namespace
