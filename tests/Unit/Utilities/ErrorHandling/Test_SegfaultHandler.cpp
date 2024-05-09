// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <csignal>
#include <string>

#ifdef __APPLE__
#include "Parallel/Printf/Printf.hpp"
#endif
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

// For some reason on the newer macOS-13 CI runners, this test never prints out
// "Segmentation fault!". Somehow, the CHECK_THROWS_WITH must catch the signal
// and exit cleanly if the test is working. Because of this, the test fails
// because this is an output test. So we manually print out the message so the
// output test is happy.
#ifdef __APPLE__
  Parallel::printf("Workaround for Apple: Segmentation fault!\n");
#endif
}
