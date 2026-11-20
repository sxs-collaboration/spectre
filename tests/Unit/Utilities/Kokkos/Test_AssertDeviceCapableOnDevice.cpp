// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <csignal>
#include <cstdio>
#include <cstdlib>

#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/TypeTraits/IsDeviceCapable.hpp"

namespace {
SPECTRE_KOKKOS_DISABLE_EXEC_CHECK
template <typename T>
KOKKOS_FUNCTION T identity(const T& x) {
  ASSERT_DEVICE_CAPABLE(T);
  return x;
}

// ctest fails a test that is killed by a signal even if its output matches the
// OutputRegex, so exit normally when `Kokkos::abort` raises SIGABRT. The name
// must differ from the handlers in the other tests because C linkage ignores
// the anonymous namespace.
extern "C" [[noreturn]] void exit_on_device_capable_abort(
    const int /*signal*/) {
  std::_Exit(EXIT_SUCCESS);
}
}  // namespace

// Only built with a device backend, see CMakeLists.txt.
// [[OutputRegex, 'tt::is_device_capable_v<T>' violated]]
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.AssertDeviceCapableOnDevice",
                  "[Utilities][Unit]") {
  OUTPUT_TEST();
  // `std::_Exit` doesn't flush output, so don't buffer it
  REQUIRE(std::setvbuf(stdout, nullptr, _IONBF, 0) == 0);
  REQUIRE(std::signal(SIGABRT, exit_on_device_capable_abort) != SIG_ERR);
  Kokkos::parallel_for(
      "TestAssertDeviceCapableOnDevice", 1, KOKKOS_LAMBDA(const int /*i*/) {
        // `double` passes the check
        static_cast<void>(identity(1.0));
        // `std::complex` gives wrong results on GPUs, so it isn't
        // device-capable
        static_cast<void>(identity(std::complex<double>{1.0, 2.0}));
      });
  Kokkos::fence();
}
