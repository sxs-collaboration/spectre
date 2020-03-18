// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "Utilities/TypeTraits/CreateHasStaticMemberVariable.hpp"

namespace {
/// [CREATE_HAS_EXAMPLE]
CREATE_HAS_STATIC_MEMBER_VARIABLE(foo)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(foo)
CREATE_HAS_STATIC_MEMBER_VARIABLE(foobar)
CREATE_HAS_STATIC_MEMBER_VARIABLE_V(foobar)
struct testing_create_has_static_member_variable {
  static constexpr size_t foo = 1;
};

static_assert(has_foo_v<testing_create_has_static_member_variable>,
              "Failed testing CREATE_HAS_STATIC_MEMBER_VARIABLE");
static_assert(has_foo_v<testing_create_has_static_member_variable, size_t>,
              "Failed testing CREATE_HAS_STATIC_MEMBER_VARIABLE");
static_assert(not has_foo_v<testing_create_has_static_member_variable, int>,
              "Failed testing CREATE_HAS_STATIC_MEMBER_VARIABLE");
static_assert(not has_foobar_v<testing_create_has_static_member_variable>,
              "Failed testing CREATE_HAS_STATIC_MEMBER_VARIABLE");
static_assert(
    not has_foobar_v<testing_create_has_static_member_variable, size_t>,
    "Failed testing CREATE_HAS_STATIC_MEMBER_VARIABLE");
/// [CREATE_HAS_EXAMPLE]

}  // namespace
