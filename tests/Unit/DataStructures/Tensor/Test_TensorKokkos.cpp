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

    // Fill on host, then copy to device
    Scalar<DataVector> scalar_host{num_points, 0.0};
    for (size_t i = 0; i < num_points; ++i) {
      get(scalar_host)[i] = 2.0 * i;
    }
    const auto scalar = copy_to_device(scalar_host);

    // Invoke pointwise tensor operations
    Scalar<Kokkos::View<double*>> result{"result", num_points};
    Kokkos::parallel_for(
        "compute", num_points, KOKKOS_LAMBDA(const int i) {
          const Scalar<double> scalar_i = make_at_index(scalar, i);
          const Scalar<double> result_i = tensor_func(scalar_i);
          set_at_index(make_not_null(&result), result_i, i);
        });

    // Copy to host and check
    const auto result_host = copy_to_host(result);
    for (size_t i = 0; i < num_points; ++i) {
      CHECK(get(result_host)[i] == 4.0 * i);
    }
  }
}
