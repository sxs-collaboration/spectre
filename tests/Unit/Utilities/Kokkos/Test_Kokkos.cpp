// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <string>

#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Kokkos/TestKernel.hpp"

namespace {
void test_kokkos() {
  Kokkos::View<double*> view("view", 10);
  Kokkos::parallel_for(
      "TestKernel", 10, KOKKOS_LAMBDA(const int i) {
        // Implementation of `kernel_square` is in a separate library to test
        // linking it in.
        view(i) = kernel_square(static_cast<double>(i));
      });
  const auto host_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host_view, view);
  for (int i = 0; i < 10; ++i) {
    CHECK(host_view(i) == static_cast<double>(i * i));
  }
}

KOKKOS_FUNCTION double checked_sqrt(const double x) {
  SPECTRE_KOKKOS_ASSERT(x >= 0.0, "Can't take the square root of " << x);
  return sqrt(x);
}

void test_kokkos_host_assert() {
  Kokkos::View<double*> view("view", 4);
  Kokkos::parallel_for(
      "TestAssert", 4, KOKKOS_LAMBDA(const int i) {
        view(i) = checked_sqrt(static_cast<double>(i * i));
      });
  const auto host_view = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host_view, view);
  for (int i = 0; i < 4; ++i) {
    CHECK(host_view(i) == static_cast<double>(i));
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      checked_sqrt(-1.0),
      Catch::Matchers::ContainsSubstring("Can't take the square root of -1."));
#endif  // SPECTRE_DEBUG
}

SPECTRE_TEST_CASE("Unit.Utilities.Kokkos",
                   "[Utilities][Unit]") {
  test_kokkos();
  test_kokkos_host_assert();
}
}  // namespace
