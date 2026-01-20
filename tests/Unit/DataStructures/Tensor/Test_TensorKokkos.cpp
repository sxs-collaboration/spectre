// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <Kokkos_Core_fwd.hpp>
#include <cstddef>

#include "DataStructures/Tensor/AtIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {

KOKKOS_FUNCTION Scalar<double> tensor_func(const Scalar<double>& input) {
  return Scalar<double>(2.0 * get(input));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Kokkos",
                  "[Unit][DataStructures]") {
  {
    INFO("Testing Kokkos pointwise tensor operation");
    const size_t num_points = 3;
    Scalar<Kokkos::View<double*>> scalar{"scalar", num_points};
    Kokkos::parallel_for(
        "fill", num_points,
        KOKKOS_LAMBDA(const int i) { get(scalar)(i) = 2.0 * i; });

    // Invoke pointwise tensor operations
    Scalar<Kokkos::View<double*>> result{"result", num_points};
    Kokkos::parallel_for(
        "compute", num_points, KOKKOS_LAMBDA(const int i) {
          const Scalar<double> scalar_i = make_at_index(scalar, i);
          const Scalar<double> result_i = tensor_func(scalar_i);
          set_at_index(make_not_null(&result), result_i, i);
        });

    // Check result
    Kokkos::fence();
    const auto result_host = Kokkos::create_mirror_view(get(result));
    Kokkos::deep_copy(result_host, get(result));
    for (size_t i = 0; i < num_points; ++i) {
      CHECK(result_host(i) == 4.0 * i);
    }
  }
}
