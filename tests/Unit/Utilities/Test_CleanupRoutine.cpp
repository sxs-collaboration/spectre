// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <exception>

#include "Utilities/CleanupRoutine.hpp"

namespace {
void test(const bool do_throw) {
  // [cleanup_routine]
  int x = 1;
  try {
    const CleanupRoutine cleanup = [&x]() { x = 3; };
    x = 2;
    if (do_throw) {
      throw std::exception{};
    }
  } catch (const std::exception&) {
    CHECK(x == 3);
  }
  CHECK(x == 3);
  // [cleanup_routine]
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.CleanupRoutine", "[Utilities][Unit]") {
  test(false);
  test(true);
}
