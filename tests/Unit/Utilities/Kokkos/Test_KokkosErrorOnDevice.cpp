// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/StdHelpers.hpp"

namespace {
KOKKOS_FUNCTION double inverse(const double x, const size_t index) {
  if (x == 0.0) {
    const std::array<int, 2> shape{{3, 4}};
    ERROR("Can't invert " << x << " at index " << index << " of " << shape);
  }
  return 1.0 / x;
}

// ctest fails a test that is killed by a signal even if its output matches the
// OutputRegex, so exit normally when `Kokkos::abort` raises SIGABRT. The name
// must differ from the handlers in the other tests because C linkage ignores
// the anonymous namespace.
extern "C" [[noreturn]] void exit_on_error_abort(const int /*signal*/) {
  std::_Exit(EXIT_SUCCESS);
}
}  // namespace

// Only built with a device backend, see CMakeLists.txt.
// [[OutputRegex, Can't invert 0.0000000000000000e.00 at index 7 of .3,4.]]
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.ErrorOnDevice", "[Utilities][Unit]") {
  OUTPUT_TEST();
  // `std::_Exit` doesn't flush output, so don't buffer it
  REQUIRE(std::setvbuf(stdout, nullptr, _IONBF, 0) == 0);
  REQUIRE(std::signal(SIGABRT, exit_on_error_abort) != SIG_ERR);
  Kokkos::parallel_for(
      "TestErrorOnDevice", 1,
      KOKKOS_LAMBDA(const int /*i*/) { static_cast<void>(inverse(0.0, 7)); });
  Kokkos::fence();
}
