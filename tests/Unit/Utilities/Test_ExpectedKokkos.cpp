// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>

#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Expected.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
enum class Failure { Negative };

KOKKOS_FUNCTION Expected<double, Failure> checked_sqrt(const double x) {
  if (x < 0.0) {
    return Unexpected{Failure::Negative};
  }
  return sqrt(x);
}

void test_on_device() {
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    const auto root = checked_sqrt(4.0);
    const auto error = checked_sqrt(-1.0);
    return std::array{root.has_value() ? 1.0 : 0.0,
                      static_cast<bool>(root) ? 1.0 : 0.0, root.value(),
                      error.has_value() ? 1.0 : 0.0,
                      error.error() == Failure::Negative ? 1.0 : 0.0};
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Expected.Kokkos", "[Unit][Utilities]") {
  test_on_device();
}
