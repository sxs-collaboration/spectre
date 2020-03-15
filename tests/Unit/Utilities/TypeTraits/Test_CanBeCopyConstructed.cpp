// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <unordered_map>
#include <vector>

#include "Utilities/NonCopyable.hpp"
#include "Utilities/TypeTraits/CanBeCopyConstructed.hpp"

static_assert(tt::can_be_copy_constructed_v<int>,
              "Failed testing type trait is_copy_constructible");
static_assert(tt::can_be_copy_constructed_v<std::vector<int>>,
              "Failed testing type trait is_copy_constructible");
static_assert(tt::can_be_copy_constructed_v<std::unordered_map<int, int>>,
              "Failed testing type trait is_copy_constructible");
static_assert(
    not tt::can_be_copy_constructed_v<std::unordered_map<int, NonCopyable>>,
    "Failed testing type trait is_copy_constructible");
