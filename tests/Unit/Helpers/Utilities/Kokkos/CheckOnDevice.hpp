// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "Utilities/Kokkos/KokkosCore.hpp"

#ifdef SPECTRE_KOKKOS
namespace TestHelpers {
/*!
 * \brief Checks that `function` gives the same results in a Kokkos kernel as
 * on the host.
 *
 * `function` must be callable on the host and the device without arguments
 * (e.g. a `KOKKOS_LAMBDA` that captures its inputs) and return a `std::array`
 * of the values to compare. Floating-point values are compared with
 * `custom_approx` (`approx` by default) rather than exactly, because device
 * code may be compiled differently (e.g. with fused multiply-adds).
 *
 * \note You must run, not just compile, the tests. nvcc can compile code
 * that will give wrong results (e.g. host standard library algorithms).
 */
template <typename Function>
void check_on_device(const Function& function,
                     const Approx& custom_approx = approx) {
  using Result = std::decay_t<decltype(function())>;
  using ValueType = typename Result::value_type;
  constexpr size_t number_of_values = std::tuple_size_v<Result>;
  const Kokkos::View<ValueType*> device_values("device_values",
                                               number_of_values);
  Kokkos::parallel_for(
      "check_on_device", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int /*unused*/) {
        const Result values = function();
        for (size_t i = 0; i < number_of_values; ++i) {
          device_values(i) = values[i];
        }
      });
  const auto host_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, device_values);
  const Result expected_values = function();
  for (size_t i = 0; i < number_of_values; ++i) {
    CAPTURE(i);
    if constexpr (std::is_floating_point_v<ValueType>) {
      CHECK(host_values(i) == custom_approx(expected_values[i]));
    } else {
      CHECK(host_values(i) == expected_values[i]);
    }
  }
}
}  // namespace TestHelpers
#endif  // SPECTRE_KOKKOS
