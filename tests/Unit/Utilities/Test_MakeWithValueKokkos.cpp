// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"
#include "Utilities/MakeWithValue.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.MakeWithValue.Kokkos", "[Unit][Utilities]") {
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    const auto vector = make_with_value<tnsr::I<double, 3>>(1.0, 3.0);
    return std::array<double, 4>{{make_with_value<double>(1.0, 2.0),
                                  get<0>(vector), get<1>(vector),
                                  get<2>(vector)}};
  });
}
