// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/ErrorHandling/DeviceMessage.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
enum class Color : uint8_t { Red = 2, Green = 5 };

template <typename... Ts>
KOKKOS_FUNCTION std::array<char, 128> stream_all(const Ts&... values) {
  DeviceMessage message{};
  (message << ... << values);
  std::array<char, 128> result{};
  for (size_t i = 0; i < message.size() and i < result.size(); ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    gsl::at(result, i) = message.c_str()[i];
  }
  return result;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Kokkos.DeviceMessage", "[Utilities][Unit]") {
  // Formats the same message in a kernel and on the host. The doubles are
  // scaled to [1, 10) exactly, so the device's `log10` and `pow` give the same
  // digits as the host's.
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    return stream_all("x = ", 1.0, ", ", -2.5, ", ", 0.15625, ", ", 0.0, ' ',
                      true, ' ', -12, ' ', size_t{34}, ' ', Color::Green, ' ',
                      std::array<int, 3>{{1, -2, 3}}, '\n',
                      std::array<double, 1>{}, "end");
  });
}
