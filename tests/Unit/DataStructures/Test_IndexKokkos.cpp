// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Helpers/Utilities/Kokkos/CheckOnDevice.hpp"
#include "Utilities/Kokkos/KokkosCore.hpp"

namespace {
// Calls every function that can be called in Kokkos kernels in a kernel
void test_on_device() {
  TestHelpers::check_on_device(KOKKOS_LAMBDA() {
    const Index<3> index{1, 2, 3};
    const Index<3> copied_index = index;
    const Index<3> other_index{std::array<size_t, 3>{{1, 2, 4}}};
    const Index<2> isotropic_index{4};
    Index<3> modified_index = index;
    modified_index[1] = 5;
    *modified_index.begin() = 6;
    size_t sum = 0;
    for (const size_t i : index) {
      sum += i;
    }
    for (size_t& i : modified_index) {
      i += 10;
    }
    return std::array<size_t, 21>{
        {index[0],
         index[2],
         index.size(),
         index.product(),
         Index<0>{}.product(),
         isotropic_index[1],
         index.slice_away(1)[1],
         *index.begin(),
         sum,
         *index.data(),
         index.indices()[2],
         *modified_index.data(),
         modified_index[1],
         modified_index[2],
         static_cast<size_t>(index == copied_index),
         static_cast<size_t>(index != other_index),
         static_cast<size_t>(index == other_index),
         collapsed_index(Index<2>{1, 2}, Index<2>{3, 4}),
         collapsed_index(Index<3>{1, 2, 3}, Index<3>{3, 4, 5}),
         expanded_index(7, Index<2>{3, 4})[1],
         expanded_index(29, Index<3>{3, 4, 5})[2]}};
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Index.Kokkos",
                  "[DataStructures][Unit]") {
  test_on_device();
}
