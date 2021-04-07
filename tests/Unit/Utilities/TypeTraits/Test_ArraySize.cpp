// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>

#include "Utilities/TypeTraits/ArraySize.hpp"

namespace {
class A {};
}  // namespace

// [array_size_example]
static_assert(tt::array_size<std::array<double, 3>>::value == 3,
              "Failed testing type trait array_size");
static_assert(tt::array_size<std::array<A, 4>>::value == 4,
              "Failed testing type trait array_size");
// [array_size_example]
