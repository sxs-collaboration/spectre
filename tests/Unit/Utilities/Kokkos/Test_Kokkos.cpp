// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/Kokkos/TestKernel.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Kokkos",
                   "[Utilities][Unit]") {
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
