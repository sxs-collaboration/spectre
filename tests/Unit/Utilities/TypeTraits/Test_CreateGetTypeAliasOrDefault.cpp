// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/NoSuchType.hpp"

namespace {
CREATE_HAS_TYPE_ALIAS(foo)
CREATE_HAS_TYPE_ALIAS_V(foo)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(foo)
CREATE_HAS_TYPE_ALIAS(foobar)
CREATE_HAS_TYPE_ALIAS_V(foobar)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(foobar)
struct testing_create_has_type_alias {
  using foo = int;
};

static_assert(
    std::is_same_v<
        get_foo_or_default_t<testing_create_has_type_alias, NoSuchType>, int>);
static_assert(
    std::is_same_v<
        get_foobar_or_default_t<testing_create_has_type_alias, NoSuchType>,
        NoSuchType>);
}  // namespace

