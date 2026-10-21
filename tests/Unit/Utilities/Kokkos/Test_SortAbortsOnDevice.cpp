// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "Utilities/Algorithm.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// ctest fails a test that is killed by a signal even if its output matches the
// OutputRegex, so exit normally when `Kokkos::abort` raises SIGABRT. The name
// must differ from the handler in Test_KokkosAssertOnDevice.cpp because C
// linkage ignores the anonymous namespace.
extern "C" [[noreturn]] void exit_on_sort_abort(const int /*signal*/) {
  std::_Exit(EXIT_SUCCESS);
}
}  // namespace

// Only built with a device backend, see CMakeLists.txt.
// [[OutputRegex, alg::sort can't be called on the device]]
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.SortAbortsOnDevice",
                  "[Utilities][Unit]") {
  OUTPUT_TEST();
  // `std::_Exit` doesn't flush output, so don't buffer it
  REQUIRE(std::setvbuf(stdout, nullptr, _IONBF, 0) == 0);
  REQUIRE(std::signal(SIGABRT, exit_on_sort_abort) != SIG_ERR);
  Kokkos::parallel_for(
      "TestSortAbortsOnDevice", 1, KOKKOS_LAMBDA(const int /*i*/) {
        std::array<int, 3> a{{3, 1, 2}};
        alg::sort(a);
      });
  Kokkos::fence();
}
