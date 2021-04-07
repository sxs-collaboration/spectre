// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

namespace {
// [CREATE_HAS_TYPE_ALIAS]
CREATE_HAS_TYPE_ALIAS(foo_alias)
CREATE_HAS_TYPE_ALIAS_V(foo_alias)
CREATE_HAS_TYPE_ALIAS(foobar_alias)
CREATE_HAS_TYPE_ALIAS_V(foobar_alias)
struct testing_create_has_type_alias {
  using foo_alias = int;
};

static_assert(has_foo_alias_v<testing_create_has_type_alias>,
              "Failed testing CREATE_HAS_TYPE_ALIAS");
static_assert(has_foo_alias_v<testing_create_has_type_alias, int>,
              "Failed testing CREATE_HAS_TYPE_ALIAS");
static_assert(not has_foo_alias_v<testing_create_has_type_alias, double>,
              "Failed testing CREATE_HAS_TYPE_ALIAS");
static_assert(not has_foobar_alias_v<testing_create_has_type_alias>,
              "Failed testing CREATE_HAS_TYPE_ALIAS");
static_assert(not has_foobar_alias_v<testing_create_has_type_alias, int>,
              "Failed testing CREATE_HAS_TYPE_ALIAS");
// [CREATE_HAS_TYPE_ALIAS]
}  // namespace
