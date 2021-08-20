// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"

namespace {
CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(foo)

struct WithFoo {
  static constexpr int foo = 3;
};

struct WithoutFoo {};

static_assert(get_foo_or_default_v<WithFoo, 7> == 3);
static_assert(get_foo_or_default_v<WithoutFoo, 7> == 7);
}  // namespace
