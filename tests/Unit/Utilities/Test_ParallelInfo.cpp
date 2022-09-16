// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Utilities/System/ParallelInfo.hpp"

namespace sys {
namespace {

void test_time(const double seconds, const std::string& expected) {
  const std::string result = sys::pretty_wall_time(seconds);
  CHECK(result == expected);
}

void test() {
  test_time(43.0, "00:00:43");
  test_time(43.9, "00:00:43");
  test_time(2589.3, "00:43:09");
  test_time(10001.5, "02:46:41");
  test_time(123456.7, "01-10:17:36");
}

SPECTRE_TEST_CASE("Unit.Utilities.ParallelInfo", "[Unit][Utilities]") {
  test();
}
}  // namespace
}  // namespace sys
