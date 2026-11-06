// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "Utilities/ErrorHandling/DeviceMessage.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
enum class Color : uint8_t { Red = 2, Green = 5 };
}  // namespace

// Prints the message in a kernel, which runs on the device in device builds.
// [[OutputRegex, x = 1\.000000000000000000e\+00, -2\.500000000000000000e\+00 1
// -12 34 5 \(1,-2,3\) end]]
SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.DeviceMessage", "[Utilities][Unit]") {
  OUTPUT_TEST();
  Kokkos::parallel_for(
      "TestDeviceMessage", 1, KOKKOS_LAMBDA(const int /*i*/) {
        DeviceMessage message{};
        message << "x = " << 1.0 << ", " << -2.5 << ' ' << true << ' ' << -12
                << ' ' << size_t{34} << ' ' << Color::Green << ' '
                << std::array<int, 3>{{1, -2, 3}} << " end\n";
      });
  Kokkos::fence();
  REQUIRE(std::fflush(stdout) == 0);
}
