// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

namespace {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(foo)
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
static_assert(std::is_same_v<tmpl::apply<get_foo_or_default<tmpl::_1, double>,
                                         testing_create_has_type_alias>,
                             int>);
}  // namespace
