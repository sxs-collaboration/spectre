// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/HasEquivalence.hpp"

namespace {
class A {};
}  // namespace

namespace TestHelpers {
class C {};

bool operator==(const C&, const C&);
}  // namespace TestHelpers

// [has_equivalence_example]
static_assert(not tt::has_equivalence<A>::value,
              "Failed testing type trait has_equivalence");
static_assert(tt::has_equivalence<TestHelpers::C>::value,
              "Failed testing type trait has_equivalence");
static_assert(tt::has_equivalence_t<TestHelpers::C>::value,
              "Failed testing type trait has_equivalence");
static_assert(tt::has_equivalence_v<TestHelpers::C>,
              "Failed testing type trait has_equivalence");
// [has_equivalence_example]
