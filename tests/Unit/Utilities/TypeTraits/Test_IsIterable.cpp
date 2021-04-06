// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <vector>

#include "Utilities/TypeTraits/IsIterable.hpp"

// [is_iterable_example]
static_assert(tt::is_iterable<std::vector<double>>::value,
              "Failed testing type trait is_iterable");
static_assert(tt::is_iterable_t<std::vector<double>>::value,
              "Failed testing type trait is_iterable");
static_assert(tt::is_iterable_v<std::vector<double>>,
              "Failed testing type trait is_iterable");
static_assert(not tt::is_iterable<double>::value,
              "Failed testing type trait is_iterable");
// [is_iterable_example]
