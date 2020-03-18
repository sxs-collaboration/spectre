// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "Utilities/TypeTraits/IsStdArrayOfSize.hpp"

/// \cond
namespace {
class D;
}  // namespace
/// \endcond

/// [is_std_array_example]
static_assert(tt::is_std_array<std::array<double, 3>>::value,
              "Failed testing type trait is_std_array");
static_assert(tt::is_std_array_t<std::array<double, 3>>::value,
              "Failed testing type trait is_std_array");
static_assert(tt::is_std_array_v<std::array<double, 3>>,
              "Failed testing type trait is_std_array");
static_assert(not tt::is_std_array<double>::value,
              "Failed testing type trait is_std_array");
static_assert(tt::is_std_array<std::array<D, 10>>::value,
              "Failed testing type trait is_std_array");
/// [is_std_array_example]
