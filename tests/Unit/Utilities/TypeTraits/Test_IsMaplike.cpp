// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <map>
#include <unordered_map>
#include <vector>

#include "Utilities/TypeTraits/IsMaplike.hpp"

/// \cond
namespace {
class C {};
class D {};
}  // namespace
/// \endcond

/// [is_maplike_example]
static_assert(tt::is_maplike<std::unordered_map<int, double>>::value,
              "Failed testing type trait is_maplike");
static_assert(tt::is_maplike_t<std::unordered_map<int, double>>::value,
              "Failed testing type trait is_maplike");
static_assert(tt::is_maplike_v<std::unordered_map<int, double>>,
              "Failed testing type trait is_maplike");
static_assert(tt::is_maplike<std::map<int, C>>::value,
              "Failed testing type trait is_maplike");
static_assert(not tt::is_maplike<std::vector<C>>::value,
              "Failed testing type trait is_maplike");
static_assert(not tt::is_maplike<D>::value,
              "Failed testing type trait is_maplike");
/// [is_maplike_example]
