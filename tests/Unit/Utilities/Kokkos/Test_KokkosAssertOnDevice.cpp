// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
KOKKOS_FUNCTION double checked_sqrt(const double x) {
  SPECTRE_KOKKOS_ASSERT(x >= 0.0, "Can't take the square root of " << x);
  return sqrt(x);
}

// ctest fails a test that is killed by a signal even if its output matches the
// OutputRegex, so exit normally when `Kokkos::abort` raises SIGABRT.
extern "C" [[noreturn]] void exit_on_abort(const int /*signal*/) {
  std::_Exit(EXIT_SUCCESS);
}
}  // namespace

// Only built with SPECTRE_DEBUG and a device backend, see CMakeLists.txt.
// [[OutputRegex, SPECTRE_KOKKOS_ASSERT failed on the device: x >= 0.0]]
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.AssertOnDevice", "[Utilities][Unit]") {
  OUTPUT_TEST();
  // `std::_Exit` doesn't flush output, so don't buffer it
  REQUIRE(std::setvbuf(stdout, nullptr, _IONBF, 0) == 0);
  REQUIRE(std::signal(SIGABRT, exit_on_abort) != SIG_ERR);
  Kokkos::parallel_for(
      "TestAssertOnDevice", 1, KOKKOS_LAMBDA(const int /*i*/) {
        static_cast<void>(checked_sqrt(-1.0));
      });
  Kokkos::fence();
}
