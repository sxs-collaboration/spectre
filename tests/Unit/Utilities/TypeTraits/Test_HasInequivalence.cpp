// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/HasInequivalence.hpp"

/// \cond
namespace {
class A {};
}  // namespace

namespace TestHelpers {
class C {};

bool operator!=(const C&, const C&);
}  // namespace TestHelpers
/// \endcond

/// [has_inequivalence_example]
static_assert(not tt::has_inequivalence<A>::value,
              "Failed testing type trait has_inequivalence");
static_assert(tt::has_inequivalence<TestHelpers::C>::value,
              "Failed testing type trait has_inequivalence");
static_assert(tt::has_inequivalence_t<TestHelpers::C>::value,
              "Failed testing type trait has_inequivalence");
static_assert(tt::has_inequivalence_v<TestHelpers::C>,
              "Failed testing type trait has_inequivalence");
/// [has_inequivalence_example]
