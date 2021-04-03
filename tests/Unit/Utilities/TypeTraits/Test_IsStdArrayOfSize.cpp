// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "Utilities/TypeTraits/IsStdArrayOfSize.hpp"

namespace {
class D;
}  // namespace

// [is_std_array_of_size_example]
static_assert(tt::is_std_array_of_size<3, std::array<double, 3>>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(tt::is_std_array_of_size_t<3, std::array<double, 3>>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(tt::is_std_array_of_size_v<3, std::array<double, 3>>,
              "Failed testing type trait is_std_array_of_size");
static_assert(not tt::is_std_array_of_size<3, double>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(not tt::is_std_array_of_size<2, std::array<double, 3>>::value,
              "Failed testing type trait is_std_array_of_size");
static_assert(tt::is_std_array_of_size<10, std::array<D, 10>>::value,
              "Failed testing type trait is_std_array_of_size");
// [is_std_array_of_size_example]
