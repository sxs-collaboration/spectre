// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <csignal>

#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

// [[OutputRegex, Segmentation fault!]]
SPECTRE_TEST_CASE("Unit.ErrorHandling.SegfaultHandler",
                  "[ErrorHandling][Unit]") {
  ERROR_TEST();
  enable_segfault_handler();
  std::raise(SIGSEGV);
}
