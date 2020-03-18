// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <iosfwd>
#include <vector>

#include "Utilities/TypeTraits/IsStreamable.hpp"

/// \cond
namespace {
class A {};

class C {};
}  // namespace

namespace TestHelpers {
class D {};

std::ostream& operator<<(std::ostream& os, const D&) noexcept;
}  // namespace TestHelpers
/// \endcond

/// [is_streamable_example]
static_assert(not tt::is_streamable<std::ostream, C>::value,
              "Failed testing type trait is_streamable");
static_assert(not tt::is_streamable_t<std::ostream, C>::value,
              "Failed testing type trait is_streamable");
static_assert(not tt::is_streamable_v<std::ostream, C>,
              "Failed testing type trait is_streamable");
static_assert(not tt::is_streamable<std::ostream, A>::value,
              "Failed testing type trait is_streamable");
static_assert(tt::is_streamable<std::ostream, TestHelpers::D>::value,
              "Failed testing type trait is_streamable");
static_assert(tt::is_streamable_v<std::ostream, std::vector<TestHelpers::D>>,
              "Failed testing type trait is_streamable");
/// [is_streamable_example]
