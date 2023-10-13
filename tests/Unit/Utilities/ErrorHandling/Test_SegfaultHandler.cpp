// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <csignal>

#include "Utilities/ErrorHandling/SegfaultHandler.hpp"


// [[OutputRegex, Segmentation fault!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.SegfaultHandler",
                  "[ErrorHandling][Unit]") {
  // Tried to make this not an OUTPUT_TEST, but it fails on all CI compilers
  // despite passing on my desktop...
  OUTPUT_TEST();
  enable_segfault_handler();
  CHECK_THROWS_WITH(std::raise(SIGSEGV),
                    Catch::Matchers::ContainsSubstring("Segmentation fault!"));
}
